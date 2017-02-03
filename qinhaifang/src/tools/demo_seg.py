#!/usr/bin/python

# --------------------------------------------------------
# Multitask Network Cascade
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# Standard module
import os
import shutil
import argparse
import time
import cv2
import numpy as np
# User-defined module
import _init_paths
import caffe
from mnc_config import cfg
from transform.bbox_transform import clip_boxes
from utils.blob import prep_im_for_blob, im_list_to_blob
from transform.mask_transform import gpu_mask_voting
#import matplotlib.pyplot as plt
from utils.vis_seg import _convert_pred_to_image, _get_voc_color_map
from PIL import Image

# VOC 20 classes
'''
CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
'''
CLASSES = ('building')

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='MNC demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default='./models/VGG16/mnc_5stage/test_maskSeg.prototxt', type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default='./vgg16_mnc_instanceSeg_iter_2000.caffemodel', type=str)
    parser.add_argument('--datapath', dest='data_path',
                        help='path to save testdata',
                        default=cfg.ROOT_DIR +'/data/VOCdevkitSDS', type=str)
    parser.add_argument('--respath', dest='res_path',
                        help='path to save test results',
                        default=cfg.ROOT_DIR +'/test_reuslts', type=str)

    args = parser.parse_args()
    return args


def prepare_mnc_args(im, net):
    # Prepare image data blob
    blobs = {'data': None}
    processed_ims = []
    im, im_scale_factors = \
        prep_im_for_blob(im, cfg.PIXEL_MEANS, cfg.TEST.SCALES[0], cfg.TRAIN.MAX_SIZE)
    processed_ims.append(im)
    print 'pre_image_size:{}'.format(im.shape[:2])
    print 'pre_im_scale_factors:{}'.format(im_scale_factors)
    blobs['data'] = im_list_to_blob(processed_ims)
    # Prepare image info blob
    im_scales = [np.array(im_scale_factors)]
    assert len(im_scales) == 1, 'Only single-image batch implemented'
    im_blob = blobs['data']
    blobs['im_info'] = np.array(
        [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
        dtype=np.float32)
    # Reshape network inputs and do forward
    net.blobs['data'].reshape(*blobs['data'].shape)
    net.blobs['im_info'].reshape(*blobs['im_info'].shape)
    forward_kwargs = {
        'data': blobs['data'].astype(np.float32, copy=False),
        'im_info': blobs['im_info'].astype(np.float32, copy=False)
    }
    return forward_kwargs, im_scales


def im_detect(im, net):
    forward_kwargs, im_scales = prepare_mnc_args(im, net)
    blobs_out = net.forward(**forward_kwargs)
    # output we need to collect:
    # 1. output from phase1'
    rois_phase1 = net.blobs['rois'].data.copy()
    #print 'rois_phase1:{}'.format(rois_phase1.shape)
    masks_phase1 = net.blobs['mask_proposal'].data[...]
    scores_phase1 = net.blobs['seg_cls_prob'].data[...]
    # 2. output from phase2
    '''
    rois_phase2 = net.blobs['rois_ext'].data[...]
    masks_phase2 = net.blobs['mask_proposal_ext'].data[...]
    scores_phase2 = net.blobs['seg_cls_prob_ext'].data[...]
    '''
    # Boxes are in resized space, we un-scale them back
    rois_phase1 = rois_phase1[:, 1:5] / im_scales[0]
    rois_phase1, _ = clip_boxes(rois_phase1, im.shape)
    masks = masks_phase1
    boxes = rois_phase1
    scores = scores_phase1
    return boxes, masks, scores


def get_vis_dict(result_box, result_mask, img_name, cls_names, vis_thresh=0.3):
    box_for_img = []
    mask_for_img = []
    cls_for_img = []
    for cls_ind, cls_name in enumerate(cls_names):
        det_for_img = result_box[cls_ind]
        seg_for_img = result_mask[cls_ind]
        #print 'det_for_img:{}'.format(det_for_img[:,-1])
        keep_inds = np.where(det_for_img[:, -1] >= vis_thresh)[0]
        for keep in keep_inds:
            box_for_img.append(det_for_img[keep])
            mask_for_img.append(seg_for_img[keep][0])
            cls_for_img.append(cls_ind + 1)
    res_dict = {'image_name': img_name,
                'cls_name': cls_for_img,
                'boxes': box_for_img,
                'masks': mask_for_img}
    return res_dict

if __name__ == '__main__':
    args = parse_args()
    test_prototxt = args.prototxt
    test_model = args.caffemodel

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(test_prototxt, test_model, caffe.TEST)

    # Warm up for the first two images
    #im = 128 * np.ones((300, 500, 3), dtype=np.float32)
    #for i in xrange(2):
    #    _, _, _ = im_detect(im, net)
    im_file = open(os.path.join(args.data_path,'val.txt'),'r')
    im_names = im_file.readlines()
    data_path = os.path.join(args.data_path,'img')
    res_path = args.res_path
    if os.path.isdir(res_path):
        shutil.rmtree(res_path)
    os.mkdir(res_path)
    for img_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        img_name = img_name.strip()
        im_name = os.path.join(img_name + '.tif')
        print 'Demo for data/demo/{}'.format(im_name)
        print os.path.join(data_path,im_name)
        gt_image = os.path.join(data_path, im_name)
        im = cv2.imread(gt_image)
        #im = img[:200,:200,:]	
	print 'im size:{}'.format(im.shape)
        start = time.time()
        boxes, masks, seg_scores = im_detect(im, net)
        #print 'boxes{},masks{},seg_scores{}'.format(boxes.shape,masks.shape,seg_scores.shape)
        print 'boxes{}'.format(boxes.shape)
        end = time.time()
        print 'forward time %f' % (end-start)
        result_mask, result_box = gpu_mask_voting(masks, boxes, seg_scores, len(CLASSES) + 1,
                                                  300, im.shape[1], im.shape[0])
        #print 'res_box{},res_mask{}'.format(result_box.shape,result_mask.shape)
        #pred_dict = get_vis_dict(result_box, result_mask, 'data/demo/' + im_name, CLASSES)
        pred_dict = get_vis_dict(result_box, result_mask, data_path + im_name, CLASSES)

        img_width = im.shape[1]
        img_height = im.shape[0]
        inst_img, cls_img = _convert_pred_to_image(img_width, img_height, pred_dict)
        color_map = _get_voc_color_map()
        target_cls_file = os.path.join(res_path, 'cls_maskSeg_' + img_name +'.jpg')
        cls_out_img = np.zeros((img_height, img_width, 3))
        for i in xrange(img_height):
            for j in xrange(img_width):
                cls_out_img[i][j] = color_map[cls_img[i][j]][::-1]
        cv2.imwrite(target_cls_file, cls_out_img)
     
        background = Image.open(gt_image)
        #boxx = (0,0,200,200)
        #background = background.crop(boxx)
        mask = Image.open(target_cls_file)
        background = background.convert('RGBA')
        mask = mask.convert('RGBA')
        superimpose_image = Image.blend(background, mask, 0.4)
        superimpose_name = os.path.join(res_path, 'final_maskSeg_' + img_name + '.jpg')
        superimpose_image.save(superimpose_name, 'JPEG')
        print superimpose_name
