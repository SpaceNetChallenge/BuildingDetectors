#!/usr/bin/env python
# encoding=gbk
"""
Convert mask to geojson format
"""

import os
import os.path
import re
import logging
import logging.config
from multiprocessing import Pool

import skimage.io as sk
import numpy as np
import scipy.io as sio

import setting
from spaceNet import geoTools as gT
import spaceNet.image_util as img_util


def process_convert_mask_to_geojson():
    """docstring for process_convert_mask_to_geojson"""
    if setting.CONVERT_RES == 1:
        label_map_file_list = os.listdir(setting.PREDICT_LABEL_MAP_DIR)
    else:
        label_map_file_list = os.listdir(setting.LABEL_MAP_DIR_4X)
    pool_size = 32
    pool = Pool(pool_size)
    case = 0
    for convert_res in pool.imap_unordered(convert_worker, label_map_file_list):
        case += 1
        if case % 100 == 0:
            logging.info('Convert {}'.format(case))
        image_id, msg = convert_res
    pool.close()
    pool.join()


def convert_worker(mat_file):
    """docstring for convert_worker"""
    try:
        if setting.CONVERT_RES == 1:
	    image_id = '_'.join(mat_file.split('.')[0].split('_')[1:])
            print('image_id{}'.format(image_id))
            mat_file = os.path.join(setting.PREDICT_LABEL_MAP_DIR, mat_file)
            mat = sio.loadmat(mat_file)
            label_map = mat['inst_img']
            building_list = img_util.create_buildinglist_from_label_map_hull(image_id, label_map)
            #print('building_list_hull={}'.format(building_list))
            #building_list = img_util.create_buildinglist_from_label_map(image_id, label_map)
            #print('building_list_ORI={}'.format(building_list))
            geojson_file = os.path.join(setting.PREDICT_PIXEL_GEO_JSON_DIR, '{}_predict.geojson'.format(image_id))
        else:
            #print('{}'.format(mat_file))
            image_id = '_'.join(mat_file.split('.')[0].split('_')[:])
            #print('{}'.format(image_id))
            mat_file = os.path.join(setting.LABEL_MAP_DIR_4X, mat_file)
            mat = sio.loadmat(mat_file)
            label_map = mat['GTinst']['Segmentation'][0][0]
            building_list = img_util.create_buildinglist_from_label_map(image_id, label_map)
            geojson_file = os.path.join(setting.PIXEL_GEO_JSON_DIR_4X, '{}_Pixel.geojson'.format(image_id))
        gT.exporttogeojson(geojson_file, building_list)
        return image_id, 'Done'
    except Exception as e:
        logging.warning('Convert Exception[{}] image_id[{}]'.format(e, image_id))
        return image_id, e


def test_geojson():
    """docstring for test_geojson"""
    label_map_file_list = os.listdir(setting.PREDICT_LABEL_MAP_DIR)
    for mat_file in label_map_file_list:
        image_id = '_'.join(mat_file.split('.')[0].split('_')[1:])
        predict_geojson_file = os.path.join(setting.PREDICT_PIXEL_GEO_JSON_DIR, '{}_predict.geojson'.format(image_id))
        image_name = os.path.join(setting.PIC_3BAND_DIR, '3band_{}.tif'.format(image_id))
        img = sk.imread(image_name, True)
        label_map = np.zeros(img.shape, dtype=np.uint8)
        label_map = img_util.create_label_map_from_polygons(gT.importgeojson(predict_geojson_file),
                label_map)
        label_img = img_util.create_label_img(img, label_map)
        save_file = os.path.join(setting.TMP_DIR, '{}_predict.png'.format(image_id))
        sk.imsave(save_file, label_img)
        truth_geojson_file = os.path.join(setting.PIXEL_GEO_JSON_DIR, '{}_Pixel.geojson'.format(image_id))
        print('{}'.format(truth_geojson_file))
        label_map = np.zeros(img.shape, dtype=np.uint8)
        print('label_map shape{}'.format(label_map.shape))
        label_map = img_util.create_label_map_from_polygons(gT.importgeojson(truth_geojson_file), label_map)
        label_img = img_util.create_label_img(img, label_map)
        save_file = os.path.join(setting.TMP_DIR, '{}_Pixel.png'.format(image_id))
        sk.imsave(save_file, label_img)

if __name__ == '__main__':
    process_convert_mask_to_geojson()
    #test_geojson()
