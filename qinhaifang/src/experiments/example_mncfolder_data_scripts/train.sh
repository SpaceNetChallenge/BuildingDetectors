#!/bin/bash
# Usage:
# ./train.sh
# Example:
# ./train.sh

set -x
set -e

export PYTHONUNBUFFERED="True"
DATASET_TRAIN=spaceNet_train
DATASET_TEST=spaceNet_val
MODEL_PATH=./model
PROTO_PATH=./config
GPU_ID=0
ITERS=100000


# path need to rectify 
MNC_PATH=/mnt/lustre/licong/caffe_folder/mnc_tmp_folder/mn_lc
DATA_PATH=/mnt/lustre/licong/train_test/spacenet/MNC_folder/MNC_simple_v1_x1data/spaceNet_data
NET_INIT=${MODEL_PATH}/mnc_instanceSeg_x1whole_iter_40000.caffemodel

sed 's,${DATA_PATH},'${DATA_PATH}',g' ${MNC_PATH}/lib/config > ${MNC_PATH}/lib/mnc_config.py
LOG="log/instanceSeg_lot.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

srun --gres=gpu:1 --partition=m40test python ${MNC_PATH}/tools/train_net.py --gpu 0 \
  --solver ${PROTO_PATH}/solver.prototxt \
  --weights ${NET_INIT} \
  --imdb ${DATASET_TRAIN} \
  --iters ${ITERS} \
  --cfg ${PROTO_PATH}/mnc_5stage.yml

set +x
