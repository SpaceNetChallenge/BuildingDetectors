#!/usr/bin/env python
# encoding=gbk

import os
import os.path
import re
import logging
import logging.config

import skimage.io as sk
import numpy as np

import setting
import spaceNet.image_util as img_util
import scipy.io as sio


def process():
    """docstring for process"""
    img_name = '013022223131_2'
    mat_file = os.path.join(setting.TMP_DIR, '{}.mat'.format(img_name))
    image_file = os.path.join(setting.TMP_DIR, '{}.tif'.format(img_name))
    mat = sio.loadmat(mat_file)
    label_map = mat['templabel']
    img = sk.imread(image_file)
    label_img = img_util.create_label_img(img, label_map)
    save_file = os.path.join(setting.TMP_DIR, '{}_temp_label.png'.format(img_name))
    sk.imsave(save_file, label_img)


if __name__ == '__main__':
    process()
