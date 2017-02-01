#!/usr/bin/env python
# Spacenet challenge
# Process predicted class map, write result to csv
# vklimkov Dec 2016

# TODO: Rotate polygons with small area
# TODO: Split polygons that have bottlenecks
# TODO: Do windowing at the overlap locations
# TODO: http://gis.stackexchange.com/questions/3739/generalisation-strategies-for-building-outlines/3756#3756

import numpy as np
import os
import sys
import argparse
from skimage import measure
import matplotlib.pyplot as plt
from skimage.measure import find_contours, approximate_polygon, \
    subdivide_polygon
import matplotlib.patches as patches
import matplotlib as mpl
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon
import gdal
from itertools import islice
from shapely.affinity import affine_transform
import math

DIM = 512
CROP_DIM = 128
OVERLAP = 0.5
OVERLAP_DIM = int(CROP_DIM * OVERLAP)
PARTS_PER_LINE = 2 * (DIM / CROP_DIM) - 1
PARTS = PARTS_PER_LINE**2
SMALL_POLYGON_AREA = 500
NOT_POLYGON_AREA = 50

ORIG_XDIM = 438
ORIG_YDIM = 406

XFACTOR = ORIG_XDIM / float(DIM)
YFACTOR = ORIG_YDIM / float(DIM)

THRESHOLD = 0.45

MIN_DOTS_NUM = 30

CONTOURS_FINDING = 0.8
APPROXIMATION_TOLERANCE = 2.0
POLYGON_EXPANSION_PIXELS = 5


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description='Process predicted image, creates geojson')
    arg_parser.add_argument('--name', required=True,
                            help='Image name (without suffix)')
    arg_parser.add_argument('--imdir', required=True, help='Image directory')
    arg_parser.add_argument('--csv', required=True,
                            help='Output csv file')
    # arg_parser.add_argument('--band3', required=True,
    #                        help='Directory with 3-band images '
    #                             'to get geo-transform from')
    arg_parser.add_argument('--to-plot', action='store_true',
                            help='Plots extracted contours')
    args = arg_parser.parse_args()
    if not os.path.isdir(args.imdir):
        raise RuntimeError('Cant open image dir %s' % args.imdir)
    # args.band3 = os.path.join(args.band3, '3band_%s.tif' % args.name)
    # if not os.path.isfile(args.band3):
    #    raise RuntimeError('Failed to open %s' % args.band3)
    return args


def get_parts_paths(name, imdir):
    parts = []
    for i in range(PARTS):
        path = os.path.join(imdir, '%s_%d.npy' % (name, i))
        if not os.path.isfile(path):
            raise RuntimeError('Image is missing %s' % path)
        parts.append(path)
    return parts


def contour2wkt_poly(contour):
    pts = []
    for pnt in contour:
        pts.append((pnt[1] * XFACTOR, pnt[0] * YFACTOR, 0))
    return pts


def contour2world_poly(contour, geo_trans):
    ul_x = geo_trans[0]
    ul_y = geo_trans[3]
    x_dist = geo_trans[1]
    y_dist = geo_trans[5]
    pts = []
    for pnt in contour:
        x_wrld = pnt[1] * x_dist + ul_x
        y_wrld = pnt[0] * y_dist + ul_y
        pts.append((x_wrld, y_wrld, 0))
    return pts


def load_image_from_parts(paths):
    img = np.zeros([DIM, DIM])
    norm = np.zeros([DIM, DIM])
    x = 0
    y = 0
    for idx, p in enumerate(paths):
        part = np.load(p)
        img[y:y+CROP_DIM, x:x+CROP_DIM] += part
        norm[y:y+CROP_DIM, x:x+CROP_DIM] += 1
        x += OVERLAP_DIM
        if x > DIM - CROP_DIM:
            x = 0
            y += OVERLAP_DIM
        if y > DIM - CROP_DIM:
            y = 0
    img /= norm
    return img


def expand_polygon(poly, expansion_pts=POLYGON_EXPANSION_PIXELS):
    center = poly.centroid.wkt
    new_coords = []
    coords = poly.exterior.coords
    for c in coords:
        # define which quater the center is
        if c[0] <= center[0] and c[1] <= center[1]:
            # first quater
            new_pnt = (c[0] - expansion_pts, c[1] - expansion_pts)
        elif c[0] >= center[0] and c[1] <= center[1]:
            # second quater
            new_pnt = (c[0] + expansion_pts, c[1] - expansion_pts)
        elif c[0] >= center[0] and c[1] >= center[1]:
            # third quater
            new_pnt = (c[0] + expansion_pts, c[1] + expansion_pts)
        else:
            # fourth quater
            new_pnt = (c[0] - expansion_pts, c[1] + expansion_pts)
        new_coords.append(new_pnt)
    poly.exterior.coords = new_coords
    return poly



def minimum_area_bounding_box(poly):
    coords = poly.exterior.coords
    edges = ((pt2[0] - pt1[0], pt2[1] - pt1[1]) for pt1, pt2 in
            zip(coords, islice(coords, 1, None)))

    def _transformed_rects():
        for dx, dy in edges:
            length = math.sqrt(dx ** 2 + dy ** 2)
            ux, uy = dx / length, dy / length
            vx, vy = -uy, ux
            transf_rect = affine_transform(poly,
                                           (ux, uy, vx, vy, 0, 0)).envelope
            yield (transf_rect, (ux, vx, uy, vy, 0, 0))

    transf_rect, inv_matrix = min(_transformed_rects(), key=lambda r: r[0].area)
    return affine_transform(transf_rect, inv_matrix)


def main():
    args = parse_args()
    if os.path.isfile(args.csv):
        outfp = open(args.csv, 'a')
    else:
        outfp = open(args.csv, 'w')
        outfp.write('ImageId,BuildingId,PolygonWKT_Pix,PolygonWKT_Geo\n')

    if not os.path.isfile(args.name):
        process_name(args.imdir, args.name, outfp, args.to_plot)
    else:
        with open(args.name, 'r') as infp:
            for name in infp:
                name = name.strip()
                print('Processing %s...' % name)
                process_name(args.imdir, name, outfp, False)

    outfp.flush()
    outfp.close()


def process_name(imdir, name, outfp, to_plot):
    parts = get_parts_paths(name, imdir)
    img = load_image_from_parts(parts)
    #if to_plot:
    #    plt.imshow(img)
    # ds3 = gdal.Open(args.band3)
    # geo_trans = ds3.GetGeoTransform()
    img_coords = np.argwhere(img > THRESHOLD)
    if len(img_coords) > MIN_DOTS_NUM:
        db = DBSCAN(eps=2.9, min_samples=25).fit(img_coords)
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print('Estimated number of clusters: %d' % n_clusters_)
        unique_labels = set(labels)
    else:
        n_clusters_ = 0
    building_id = 1
    if n_clusters_ == 0:
        outfp.write('%s,-1,POLYGON EMPTY,POLYGON EMPTY\n' % name)
    else:
        for k in unique_labels:
            if k == -1:
                continue
            class_member_mask = (labels == k)
            xy = img_coords[class_member_mask]
            cluster_img = np.zeros(img.shape)
            #if to_plot:
            #    plt.plot(xy[:, 1], xy[:, 0])
            cluster_img[xy[:, 0], xy[:, 1]] = 1.0
            # to ensure closed contour
            cluster_img[0, :] = 0.0
            cluster_img[DIM - 1, :] = 0.0
            cluster_img[:, 0] = 0.0
            cluster_img[:, DIM - 1] = 0.0
            contours = measure.find_contours(cluster_img, CONTOURS_FINDING)
            contour = contours[0]
            contour = approximate_polygon(contour,
                                          tolerance=APPROXIMATION_TOLERANCE)
            if len(contour) < 3:
                continue
            exterior_wkt = contour2wkt_poly(contour)
            # exterior_wrld = contour2world_poly(contour, geo_trans)
            if to_plot:
                plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
            interior_wkt = []
            # interior_wrld = []
            if len(contours) > 1:
                print('**WARNING! more than one contour created for cluster')
                # Assuming inner contours
                for contour in contours[1:2]:
                    contour = approximate_polygon(contour,
                                                  tolerance=APPROXIMATION_TOLERANCE)
                    if len(contour) < 3:
                        continue
                    interior_wkt.append(contour2wkt_poly(contour))
                    # interior_wrld.append(contour2world_poly(contour, geo_trans))
                    if to_plot:
                        plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
            wkt_plgn = Polygon(exterior_wkt, interior_wkt)
            if wkt_plgn.area <= NOT_POLYGON_AREA:
                continue
            wkt_plgn = minimum_area_bounding_box(wkt_plgn)
            wkt_plgn = expand_polygon(wkt_plgn)
            # Not using polygon in world coordinates for now
            # wrld_plgn = Polygon(exterior_wrld, interior_wrld)
            line = '%s,%d,\"%s\", 1\n' % (name, building_id, str(wkt_plgn))
            line = line.replace(' Z ', ' ')
            line = line.replace(', ', ',')
            outfp.write(line)
            building_id += 1
        if to_plot:
            plt.gca().set_xlim([0, DIM])
            plt.gca().set_ylim([0, DIM])
            plt.gca().invert_yaxis()
            plt.show()


if __name__ == '__main__':
    main()
