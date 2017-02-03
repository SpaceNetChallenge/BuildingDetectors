#!/bin/sh

# path need to rectify
MNC_PATH=/mnt/lustre/licong/caffe_folder/MNC_lc


srun --gres=gpu:1 python ${MNC_PATH}/tools/demo_seg.py \
--net=./model/test_vgg16_mnc_instanceSeg_iter_11000.caffemodel \
--datapath=./testdata \
--def=./config/test_instanceSeg.prototxt \
--respath=./testdata/results_x1_11k
