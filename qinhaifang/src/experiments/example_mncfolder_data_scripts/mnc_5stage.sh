#!/bin/bash
# Usage:
# ./experiments/scripts/mnc_5stage.sh GPU NET [--set ...]
# Example:
# ./experiments/scripts/mnc_5stage.sh 0 VGG16 \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400,500,600,700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
ITERS=100000
MNC_PATH=/mnt/lustre/licong/caffe_folder/mnc_tmp_folder/mnc_lc
DATA_PATH=/mnt/lustre/licong/train_test/spacenet/MNC_folder/MNC_simple_v1_x1data/spaceNet_data
sed 's,${DATA_PATH},'${DATA_PATH}',g' ${MNC_PATH}/lib/config > ${MNC_PATH}/lib/mnc_config.py


DATASET_TRAIN=spaceNet_train
DATASET_TEST=spaceNet_val
MODEL_PATH=./model
PROTO_PATH=./config



LOG="log/instanceSeg_lot.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

NET_INIT=${MODEL_PATH}/VGG16.mask.caffemodel
time ${MNC_PATH}/tools/train_net.py --gpu ${GPU_ID} \
  --solver ${PROTO_PATH}/solver.prototxt \
  --weights ${NET_INIT} \
  --imdb ${DATASET_TRAIN} \
  --iters ${ITERS} \
  --cfg ${PROTO_PATH}/mnc_5stage.yml \
  
set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def ${MNC_PATH}/tools/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${DATASET_TEST} \
  --cfg ${PROTO_PATH}/mnc_5stage.yml \
  --task seg

