#!/usr/bin/env python
# Spacenet challenge
# Creates input and target data for training.
# Stores numpy matrices. Target matrix is reshaped to train cross-entropy.
# vklimkov Dec 2016

import json
from shapely.geometry import Polygon, shape, Point
import gdal
import numpy as np
import os
import cv2
import argparse

MAX_UINT8 = 255.0
MAX_UINT16 = 65535.0

ORIG_XDIM = 438
ORIG_YDIM = 406

EXPECTED_DIM = 512

XFACTOR = EXPECTED_DIM / float(ORIG_XDIM)
YFACTOR = EXPECTED_DIM / float(ORIG_YDIM)


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description='creates in/out features for spacenet CNN training')
    arg_parser.add_argument('--band3', required=True,
                            help='Path to 3band image')
    arg_parser.add_argument('--band8', required=True,
                            help='Path to 8band image')
    arg_parser.add_argument('--geo', help='Path to geo')
    arg_parser.add_argument('--in-dir', required=True,
                            help='Directory to put DNN in files to')
    arg_parser.add_argument('--target-dir',
                            help='Directory to put DNN target files to')
    arg_parser.add_argument('--only-infeats', action='store_true', help='')
    arg_parser.add_argument('--in-channels', type=list, action='store',
                            default=[0, 1, 2, 10])
    args = arg_parser.parse_args()
    if not os.path.isfile(args.band3):
        raise RuntimeError('Failed to open 3band image %s' % args.band3)
    if not os.path.isfile(args.band8):
        raise RuntimeError('Failed to open 8band image %s' % args.band8)
    if not os.path.isdir(args.in_dir):
        os.makedirs(args.in_dir)
    if not args.only_infeats:
        if not os.path.isdir(args.target_dir):
            os.makedirs(args.target_dir)
        if not os.path.isfile(args.geo):
            raise RuntimeError('Failed to open geo %s' % args.geo)
    args.name = os.path.splitext(os.path.basename(args.band3))[0]
    _, args.name = args.name.split('_', 1)
    return args


def world_2_pixel(geo_trans, i, j):
    ul_x = geo_trans[0]
    ul_y = geo_trans[3]
    x_dist = geo_trans[1]
    y_dist = geo_trans[5]
    x_pix = (i - ul_x) / x_dist
    y_pix = (j - ul_y) / y_dist
    return[round(x_pix), round(y_pix)]


def convert_points(points, geo_trans):
    converted_points = []
    for p in points:
        cp = Point(world_2_pixel(geo_trans, p[0], p[1]))
        converted_points.append([cp.x, cp.y])
    return converted_points


def draw_polygon(polygon, geo_trans, buildings, borders):
    points = polygon.exterior.coords[:]
    converted_points = convert_points(points, geo_trans)
    cv2.drawContours(buildings,
                     [np.array(converted_points, dtype=np.int32)],
                     -1, 1, thickness=-1)
    cv2.drawContours(borders,
                     [np.array(converted_points, dtype=np.int32)],
                     -1, 1, thickness=2)
    if polygon.interiors:
        for inner_polygon in polygon.interiors:
            points = inner_polygon.coords[:]
            converted_points = convert_points(points, geo_trans)
            cv2.drawContours(buildings,
                             [np.array(converted_points, dtype=np.int32)],
                             -1, 0, thickness=-1)
            cv2.drawContours(borders,
                             [np.array(converted_points, dtype=np.int32)],
                             -1, 1, thickness=2)


def main():
    args = parse_args()
    ds3 = gdal.Open(args.band3)
    ds8 = gdal.Open(args.band8)
    if not args.only_infeats:
        assert(ds3.RasterXSize == ds8.RasterXSize == EXPECTED_DIM)
        assert (ds3.RasterYSize == ds8.RasterYSize == EXPECTED_DIM)
        assert(ds3.RasterCount == 3)
        assert(ds8.RasterCount == 8)
        geo_trans = ds3.GetGeoTransform()
        borders = np.zeros([EXPECTED_DIM, EXPECTED_DIM])
        building = np.zeros([EXPECTED_DIM, EXPECTED_DIM])

        with open(args.geo, 'r') as f:
            js = json.load(f)
            for feature in js['features']:
                polygon = shape(feature['geometry'])
                if polygon.type == 'Polygon':
                    draw_polygon(polygon, geo_trans, building, borders)
                elif polygon.type == 'MultiPolygon':
                    draw_polygon(polygon[0], geo_trans, building, borders)
                else:
                    raise RuntimeError('unknown polygom type!')

        # save target files
        building -= borders
        building = np.clip(building, 0, 1)
        street = np.zeros([EXPECTED_DIM, EXPECTED_DIM])
        street.fill(1)
        street -= building
        target = np.zeros([EXPECTED_DIM, EXPECTED_DIM, 2])
        target[:, :, 0] = street
        target[:, :, 1] = building
        target = np.reshape(target, (EXPECTED_DIM * EXPECTED_DIM, 2))
        np.save(os.path.join(args.target_dir, 'target_%s' % args.name), target)

    # normalize and save input files for training
    inputs = np.zeros([EXPECTED_DIM, EXPECTED_DIM, 3 + 8])
    input_idx = 0
    for i in range(1, 4):
        channel = np.array(ds3.GetRasterBand(i).ReadAsArray()) / MAX_UINT8
        inputs[:, :, input_idx] = channel
        input_idx += 1
    for i in range(1, 9):
        channel = np.array(ds8.GetRasterBand(i).ReadAsArray()) / MAX_UINT16
        inputs[:, :, input_idx] = channel
        input_idx += 1

    # limit with the channels specified
    inputs = inputs[:, :, args.in_channels]
    np.save(os.path.join(args.in_dir, 'in_%s' % args.name), inputs)


if __name__ == '__main__':
    main()
