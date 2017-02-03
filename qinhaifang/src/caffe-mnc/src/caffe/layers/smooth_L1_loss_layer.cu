// --------------------------------------------------------
// Multitask Network Cascade
// Modified from caffe-fast-rcnn (https://github.com/rbgirshick/caffe-fast-rcnn)
// Copyright (c) 2016, Haozhi Qi
// Licensed under The MIT License [see LICENSE for details]
// --------------------------------------------------------


#include "caffe/fast_rcnn_layers.hpp"
#include <iostream>
namespace caffe {

template <typename Dtype>
__global__ void SmoothL1Forward(const int n, const Dtype* in, Dtype* out,
    Dtype sigma2) {
  // f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
  //        |x| - 0.5 / sigma / sigma    otherwise
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = in[index];
    Dtype abs_val = abs(val);
    if (abs_val < 1.0 / sigma2) {
      out[index] = 0.5 * val * val * sigma2;
    } else {
      out[index] = abs_val - 0.5 / sigma2;
    }
  }
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());    // d := b0 - b1
  if (has_weights_) {
    // apply "inside" weights
    caffe_gpu_mul(
        count,
        bottom[2]->gpu_data(),
        diff_.gpu_data(),
        diff_.mutable_gpu_data());  // d := w_in * (b0 - b1)
  }
  SmoothL1Forward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, diff_.gpu_data(), errors_.mutable_gpu_data(), sigma2_);
  CUDA_POST_KERNEL_CHECK;

  if (has_weights_) {
    // apply "outside" weights
    caffe_gpu_mul(
        count,
        bottom[3]->gpu_data(),
        errors_.gpu_data(),
        errors_.mutable_gpu_data());  // d := w_out * SmoothL1(w_in * (b0 - b1))
  }

  Dtype loss;
  caffe_gpu_dot(count, ones_.gpu_data(), errors_.gpu_data(), &loss);
  top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
}

template <typename Dtype>
__global__ void SmoothL1Backward(const int n, const Dtype* in, Dtype* out,
    Dtype sigma2) {
  // f'(x) = sigma * sigma * x         if |x| < 1 / sigma / sigma
  //       = sign(x)                   otherwise
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = in[index];
    Dtype abs_val = abs(val);
    if (abs_val < 1.0 / sigma2) {
      out[index] = sigma2 * val;
    } else {
      out[index] = (Dtype(0) < val) - (val < Dtype(0));
    }
  }
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // after forwards, diff_ holds w_in * (b0 - b1)
  int count = diff_.count();
  SmoothL1Backward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, diff_.gpu_data(), diff_.mutable_gpu_data(), sigma2_);
  CUDA_POST_KERNEL_CHECK;
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          count,                           // count
          alpha,                           // alpha
          diff_.gpu_data(),                // x
          Dtype(0),                        // beta
          bottom[i]->mutable_gpu_diff());  // y
      if (has_weights_) {
        // Scale by "inside" weight
        caffe_gpu_mul(
            count,
            bottom[2]->gpu_data(),
            bottom[i]->gpu_diff(),
            bottom[i]->mutable_gpu_diff());
        // Scale by "outside" weight
        caffe_gpu_mul(
            count,
            bottom[3]->gpu_data(),
            bottom[i]->gpu_diff(),
            bottom[i]->mutable_gpu_diff());
      }
    }
  }
  Dtype* bottom_data = bottom[0]->mutable_cpu_data();
  Dtype* bottom_data2 = bottom[1]->mutable_cpu_data();
  Dtype* diff_print = diff_.mutable_cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // we print the gradient for RPN loss
  /*
  if (bottom[0]->shape()[0] == 1) {
    LOG(INFO) << "SmoothL1Debug: " << bottom[0]->shape_string();
    LOG(INFO) << "SmoothL1Debug: " << bottom[0]->shape()[3] << " " << bottom[0]->shape()[2] << " " << bottom[0]->shape()[1];
    LOG(INFO) << top[0]->cpu_diff()[0] << " " << bottom[0]->num();
    
    for (int w = 0; w < bottom[0]->shape()[3]; w++) {
      for (int h = 0; h < bottom[0]->shape()[2]; h++) {
        for (int c = 0; c < bottom[0]->shape()[1]; c++) {
          Dtype data = bottom_diff[c*bottom[0]->shape()[2]*bottom[0]->shape()[3] + h * bottom[0]->shape()[3] + w];
          if (data < 1e-12 && data > -1 * 1e-12) continue;
          std::cout << "(" << c << "," << h << "," << w << "): " << data << std::endl;
        }
      }
    }
  }
  */
}

INSTANTIATE_LAYER_GPU_FUNCS(SmoothL1LossLayer);

}  // namespace caffe
