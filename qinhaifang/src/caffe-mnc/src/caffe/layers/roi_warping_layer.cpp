// --------------------------------------------------------
// Multitask Network Cascade
// Written by Haozhi Qi
// Copyright (c) 2016, Haozhi Qi
// Licensed under The MIT License [see LICENSE for details]
// --------------------------------------------------------

#include <cfloat>

#include "caffe/fast_rcnn_layers.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void ROIWarpingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ROIWarpingParameter roi_warping_param = this->layer_param_.roi_warping_param();
  CHECK_GT(roi_warping_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(roi_warping_param.pooled_w(), 0)
      << "pooled_w must be > 0";

  pooled_height_ = roi_warping_param.pooled_h();
  pooled_width_ = roi_warping_param.pooled_w();
  spatial_scale_ = roi_warping_param.spatial_scale();
}

template <typename Dtype>
void ROIWarpingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_);
  max_idx_w_.Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_);
  max_idx_h_.Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_);
  buffer_.Reshape(bottom[1]->num() * 5, channels_, pooled_height_, pooled_width_);
}

template <typename Dtype>
void ROIWarpingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void ROIWarpingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(ROIWarpingLayer);
#endif

INSTANTIATE_CLASS(ROIWarpingLayer);
REGISTER_LAYER_CLASS(ROIWarping);

}  // namespace caffe
