#!/usr/bin/env python
#encoding=gb18030

"""
image_util
"""

from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import skimage.draw as sk_draw  # use skimage to create label_map;
import skimage.measure as sk_measuer
import skimage
import numpy as np
import shapely.geometry
import shapely.wkt
from spaceNet import geoTools as gT


def create_label_map_from_polygons(building_list, label_map):
    """ create label map from polygons
    Dependencies: skimage
    Input:
        polygons: same as the output of importgeojson
    Output:
        label_map: 2-D ndarray
    """
    for building in building_list:
        polygon = building['poly']
        ring = polygon.GetGeometryRef(0)
        xx, yy = [], []
        for i in range(0, ring.GetPointCount()):
            y, x, z = ring.GetPoint(i)
            xx.append(x)
            yy.append(y)
        xx = np.array(xx)
        yy = np.array(yy)
        rr, cc = sk_draw.polygon(xx, yy)
        #print('{}, {}'.format(rr, cc))
        label_map[rr, cc] = building['BuildingId']
    return label_map

def create_buildinglist_from_label_map(image_id, label_map):
    """docstring for create_buildinglist_from_label_map"""
    building_list = []
    building_id = 1
    max_label = np.max(np.max(label_map))
    for i in range(1, max_label + 1):
        tmp_label_map = np.zeros(label_map.shape, dtype=np.uint8)
        x, y = np.nonzero(label_map == i)
        tmp_label_map[x, y] = 1
        contours = sk_measuer.find_contours(tmp_label_map, 0)
        #print('contours:{}'.format(contours))
        if len(contours) == 0:
            continue
        contour = contours[0]
        #print('contour:{}'.format(contour))
        # Create ring
        ring = ogr.Geometry(ogr.wkbLinearRing)
        pt_num = contour.shape[0]
        for i in range(0, pt_num):
            ring.AddPoint(int(contour[i][1]), int(contour[i][0])) #(x, y, 0)
        # Create polygon
        #print('ring:{}'.format(ring))
        ring.CloseRings()
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        #print('poly:{}'.format(poly))
        building_list.append({'ImageId': image_id, 'BuildingId': building_id, 'poly': poly})
        building_id += 1
        print('Done')
    return building_list

def create_buildinglist_from_label_map_hull(image_id, label_map):
    """docstring for create_buildinglist_from_label_map"""
    building_list = []
    building_id = 1
    counts = np.bincount(label_map.flatten())
    for i, size in enumerate(counts):
        if 20000 >= size and size >= 20:
            points = shapely.geometry.MultiPoint([(y, x, 0) for x,y in np.argwhere(label_map == i)])
            hull = points.convex_hull
            #print('imageid={},hull={}'.format(image_id,hull))
            poly_odi = shapely.wkt.dumps(hull, old_3d = True, rounding_precision = 2)
            #print('imageid={},poly={}'.format(image_id,poly_odi))
            ring = ogr.Geometry(ogr.wkbLinearRing)
            point = ogr.CreateGeometryFromWkt(poly_odi)
            #print('point={}'.format(point))
            #ring.AddPoint(point)
            #poly = ogr.Geometry(ogr.wkbPolygon)
            #poly.AddGeometry(ring)
            building_list.append({'ImageId': image_id, 'BuildingId': building_id, 'poly': point})
            building_id += 1
            #print('Done')
    return building_list

def create_label_img(img, label_map):
    """docstring for create_label_img"""
    colors = ['red', 'yellow', 'blue', 'green', 'magenta', 'aquamarine', 'purple']
    #colors = ['blue']
    new_img = skimage.color.label2rgb(label_map, img, colors, alpha=0.2, bg_label=0)
    return new_img
