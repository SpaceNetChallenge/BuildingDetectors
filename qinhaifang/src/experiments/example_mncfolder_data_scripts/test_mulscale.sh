#!/bin/sh
MNC_PATH=/mnt/lustre/licong/caffe_folder/mnc_tmp_folder/mnc_lc

srun --gres=gpu:1 python ${MNC_PATH}/tools/demo_seg_mulscale.py \
--net=./model/mnc_instanceSeg_x1whole_iter_38000.caffemodel \
--datapath=./testdata/x1_data \
--def=./config/ori_test_instanceSeg.prototxt \
--datalist=val4.txt \
--respath=./testdata/temp_val4_2

