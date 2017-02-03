#!/bin/sh

srun --gres=gpu:1 ./experiments/scripts/mnc_5stage.sh 0 VGG16

