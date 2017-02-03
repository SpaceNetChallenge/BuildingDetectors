#!/usr/bin/env python
#encoding=gb18030

"""
setting for some folder
"""

import os
import os.path

CONVERT_RES = 1
LOGGING_CONF_FILE = "conf/log.conf"
DATA_DIR = '/data/building_extraction/SpaceNet/data'
TMP_DIR = 'tmp'
LOG_DIR = 'log'
PIC_3BAND_DIR = os.path.join(DATA_DIR, '3band_public')
GEO_JSON_DIR = os.path.join(DATA_DIR, 'geoJson')
LABEL_MAP_DIR = os.path.join(DATA_DIR, 'label_map')
LABEL_MAP_DIR_4X = os.path.join(DATA_DIR, 'inst')
PIXEL_GEO_JSON_DIR = os.path.join(DATA_DIR, 'pixelGeoJson')
PIXEL_GEO_JSON_DIR_4X = os.path.join(DATA_DIR, 'pixelGeoJson_4x')
RAW_LABEL_IMG_DIR = os.path.join(DATA_DIR, 'raw_label_image')
PREDICT_LABEL_MAP_DIR = os.path.join(DATA_DIR, 'predict_label_map')
PREDICT_PIXEL_GEO_JSON_DIR = os.path.join(DATA_DIR, 'predict_pixelGeoJson')
