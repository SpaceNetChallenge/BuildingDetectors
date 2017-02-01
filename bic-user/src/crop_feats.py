#!/usr/bin/env python
# Spacenet challenge
# Crops feats in smaller overlapping images
# vklimkov Dec 2016

import numpy as np
import os
import argparse
import glob

EXPECTED_DIM = 512
CROP_DIM = 128
OVERLAP = 0.5


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description='Crops features in given dir')
    arg_parser.add_argument('--infeat', required=True, help='Input feature')
    arg_parser.add_argument('--outdir', required=True, help='Output dir')
    arg_parser.add_argument('--to-reshape', action='store_true')
    args = arg_parser.parse_args()
    if not os.path.isfile(args.infeat):
        raise RuntimeError('Cant open input feat %s' % args.infeat)
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)
    return args


def crop_image(im):
    pieces = []
    start_pts = np.linspace(0, EXPECTED_DIM,
                            EXPECTED_DIM / (OVERLAP * CROP_DIM) + 1)
    start_pts = start_pts[:-2]
    start_pts = [int(x) for x in start_pts]
    for i in start_pts:
        for j in start_pts:

            croped_image = im[i:i+CROP_DIM, j:j+CROP_DIM, :]
            pieces.append(croped_image)
    return pieces


def main():
    args = parse_args()
    feat = np.load(args.infeat)
    dims = feat.shape
    if args.to_reshape:
        assert(len(dims) == 2)
        assert(dims[0] == EXPECTED_DIM * EXPECTED_DIM)
        feat = np.reshape(feat, (EXPECTED_DIM, EXPECTED_DIM, dims[1]))
    else:
        assert(len(dims) == 3)
        assert(dims[0] == dims[1] == EXPECTED_DIM)
    pieces = crop_image(feat)
    name = os.path.splitext(os.path.basename(args.infeat))[0]
    for idx in range(len(pieces)):
        piece = pieces[idx]
        if args.to_reshape:
            piece = np.reshape(piece, (CROP_DIM * CROP_DIM, dims[1]))
        np.save(os.path.join(args.outdir, '%s_%d' % (name, idx)), piece)


if __name__ == '__main__':
    main()