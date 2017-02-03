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
from spaceNet import geoTools as gT
import spaceNet.image_util as img_util


def test():
    """docstring for test"""
    image_id = 'AOI_1_RIO_img1146'
    image_file = os.path.join(setting.PIC_3BAND_DIR, '3band_{}.tif'.format(image_id))
    wgs_geojson_file = os.path.join(setting.GEO_JSON_DIR, 'Geo_{}.geojson'.format(image_id))
    print(wgs_geojson_file)
    print(image_file)
    pixel_geojson_file = './test/013022223130_Public_img104_pixel.geojson'
    building_list = gT.convert_wgs84geojson_to_pixgeojson(wgs_geojson_file,
            image_file, image_id, pixel_geojson_file)
    for building in building_list:
        print(building)


if __name__ == '__main__':
    test()
