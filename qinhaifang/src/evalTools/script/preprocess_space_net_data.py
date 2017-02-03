#!/usr/bin/env python
# encoding=gbk

import os
import os.path
import re
import logging
import logging.config
from multiprocessing import Pool

import skimage.io as sk
import numpy as np

import setting
from spaceNet import geoTools as gT
import spaceNet.image_util as img_util


class PreprocessSpaceNetData(object):
    """docstring for PreprocessSpaceNetData"""
    def __init__(self):
        self._logger = logging.getLogger()

    def _convert_to_pixgeojson(self):
        """docstring for _convert_to_pixgeojson"""
        img_list = os.listdir(setting.PIC_3BAND_DIR)
        pool_size = 8
        pool = Pool(pool_size)
        case = 0
        suc_num = 0
        fail_num = 0
        fail_list = []
        for res in pool.imap_unordered(convert_wgs84geojson_to_pixgeojson_worker, img_list):
            case += 1
            if case % 200 == 0:
                self._logger.info('Processed {}'.format(case))
            image_id, convert_msg = res
            if convert_msg == 'Done':
                suc_num += 1
            else:
                fail_num += 1
                fail_list.append([image_id, convert_msg])
        self._logger.info('Convert Done. Suc[{}] Fail[{}]'.format(suc_num, fail_num))
        pool.close()
        pool.join()
        # output fail image_ids
        fail_images_file = os.path.join(setting.TMP_DIR, 'fail_imgs')
        with open(fail_images_file, 'w') as fout:
            for image_id, convert_msg in fail_list:
                fout.write('{}\t{}'.format(image_id, convert_msg))

    def _generate_label_map(self):
        """docstring for _generate_label_map"""
        img_list = os.listdir(setting.PIC_3BAND_DIR)
        pool_size = 8
        pool = Pool(pool_size)
        case = 0
        for gene_res in pool.imap_unordered(generate_label_map_worker, img_list):
            case += 1
            if case % 200 == 0:
                self._logger.info('Generate_Label_Map {}'.format(case))
        pool.close()
        pool.join()

    def save_raw_label_img(self):
        """docstring for test_blending"""
        img_list = os.listdir(setting.PIC_3BAND_DIR)
        pool_size = 8
        pool = Pool(pool_size)
        case = 0
        for gene_res in pool.imap_unordered(save_raw_label_img_worker, img_list):
            case += 1
            if case % 200 == 0:
                self._logger.info('RawLabelImage {}'.format(case))
        pool.close()
        pool.join()

    def process(self):
        """docstring for process"""
        self._convert_to_pixgeojson()
        #self._generate_label_map()
        #self.save_raw_label_img()


def save_raw_label_img_worker(image_name):
    """docstring for save_raw_label_img_worker
    """
    try:
        image_id = '_'.join(image_name.split('.')[0].split('_')[1:])
        image_name = os.path.join(setting.PIC_3BAND_DIR, image_name)
        label_map_file = os.path.join(setting.LABEL_MAP_DIR,
                '{}.npy'.format(image_id))
        img = sk.imread(image_name)
        label_map = np.load(label_map_file)
        label_img = img_util.create_label_img(img, label_map)
        save_file = os.path.join(setting.RAW_LABEL_IMG_DIR, '{}_raw_label.png'.format(image_id))
        sk.imsave(save_file, label_img)
        return image_id, 'Done'
    except Exception as e:
        logging.warning('Generate_Label_Map Exception[{}] image_id[{}]'.format(e,
            image_id))
        return image_id, e


def generate_label_map_worker(image_name):
    """docstring for generate_label_map_worker"""
    try:
        image_id = '_'.join(image_name.split('.')[0].split('_')[1:])
        image_name = os.path.join(setting.PIC_3BAND_DIR, image_name)
        pixel_geojson_file = os.path.join(setting.PIXEL_GEO_JSON_DIR,
                '{}_Pixel.geojson'.format(image_id))
        label_map_file = os.path.join(setting.LABEL_MAP_DIR,
                '{}.npy'.format(image_id))
        img = sk.imread(image_name, True)
        label_map = np.zeros(img.shape, dtype=np.uint8)
        img_util.create_label_map_from_polygons(gT.importgeojson(pixel_geojson_file),
                label_map)
        np.save(label_map_file, label_map)
        return image_id, 'Done'
    except Exception as e:
        logging.warning('Generate_Label_Map Exception[{}] image_id[{}]'.format(e,
            image_id))
        return image_id, e


def convert_wgs84geojson_to_pixgeojson_worker(image_name):
    """docstring for convert_wgs84geojson_to_pixgeojson_worker"""
    try:
        image_id = '_'.join(image_name.split('.')[0].split('_')[1:])
        image_name = os.path.join(setting.PIC_3BAND_DIR, image_name)
        wgs_geojson_file = os.path.join(setting.GEO_JSON_DIR,
                '{}_Geo.geojson'.format(image_id))
        pixel_geojson_file = os.path.join(setting.PIXEL_GEO_JSON_DIR,
                '{}_Pixel.geojson'.format(image_id))
        building_list = gT.convert_wgs84geojson_to_pixgeojson(wgs_geojson_file,
                image_name, image_id, pixel_geojson_file)
        return image_id, 'Done'
    except Exception as e:
        logging.warning('Convert wgs_2_pix Exception[{}] image_id[{}]'.format(e,
            image_id))
        return image_id, e
