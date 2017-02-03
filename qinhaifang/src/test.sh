#!/bin/sh
srun --gres=gpu:1 --nodelist=BJ-IDC1-10-10-20-39 python ./tools/demo_seg.py --net=vgg16_mnc_instanceSeg_iter_17000.caffemodel

#--nodelist=BJ-IDC1-10-10-20-27
