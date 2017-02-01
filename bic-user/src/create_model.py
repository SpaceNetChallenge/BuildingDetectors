#!/usr/bin/env python
# Spacenet challenge
# Creates and trains CNN to recgonize buildings
# vklimkov Dec 2016

import argparse
import os
import sys
import numpy as np
import glob
from random import shuffle
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import TimeDistributed
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import adam
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Deconvolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.utils import np_utils
from keras import backend as K

# fix random seed for reproducibility
seed = 42
np.random.seed(seed)

DIM = 128
CHANNELS = 4
CLASSES = 2
BATCHSIZE = 16
# this was calculated from the assumtion:
# 776kb per training sample * 3 augmentation moves = 2.273 mB per sample
# 12 gB RAM on GPU. (200)
DATAGEN_PERSTEP = BATCHSIZE * 80
EPOCHS = 100
SAVE_MODEL_EVERY_NTH_EPOCH = 10
# just vertical and horizontal flip
AUGMENTATION = 3
LEARNING_RATE = 0.0001


class DataGenerator:
    def __init__(self, indir, targetdir, perstep=DATAGEN_PERSTEP,
                 minibatch=BATCHSIZE):
        self.train_idx = 0
        self.perstep = perstep
        self.minibatch = minibatch
        print('training step %d' % self.perstep)
        self.indir = indir
        self.targetdir = targetdir
        # some sanity checks
        innames = glob.glob('%s/*.npy' % indir)
        # dont forget to remove "in_" from names
        innames = [os.path.splitext(os.path.basename(x))[0][3:]
                   for x in innames]
        targetnames = glob.glob('%s/*.npy' % targetdir)
        # dont forget to remove "target_" from names
        targetnames = [os.path.splitext(os.path.basename(x))[0][7:]
                       for x in targetnames]
        if len(innames) != len(targetnames):
            raise RuntimeError('different amount of in and target files')
        targetnames = set(targetnames)
        for i in innames:
            if i not in targetnames:
                raise RuntimeError('%s is missing from target directory' % i)
        shuffle(innames)
        self.train_names = innames
        print('total num of training imgs %d' % len(self.train_names))

    def flow_training(self):
        train_to = min(len(self.train_names), self.train_idx + self.perstep)
        samples = (train_to - self.train_idx) * AUGMENTATION
        x_train = np.zeros((samples, DIM, DIM, CHANNELS))
        y_train = np.zeros((samples, DIM * DIM, CLASSES))
        sample_idx = 0
        for i in range(self.train_idx, train_to):
            name = self.train_names[i]
            x = np.load(os.path.join(self.indir, 'in_%s.npy' % name))
            y = np.load(os.path.join(self.targetdir, 'target_%s.npy' % name))
            x_train[sample_idx, :, :, :] = x
            y_train[sample_idx, :, :] = y
            sample_idx += 1
            # augmentation
            yy = np.reshape(y, (DIM, DIM, CLASSES))
            # left-to-right mirror
            x_aug = np.fliplr(x)
            y_aug = np.fliplr(yy)
            y_aug = np.reshape(y_aug, (DIM * DIM, CLASSES))
            x_train[sample_idx, :, :, :] = x_aug
            y_train[sample_idx, :, :] = y_aug
            sample_idx += 1
            # up to down mirror
            x_aug = np.flipud(x)
            y_aug = np.flipud(yy)
            y_aug = np.reshape(y_aug, (DIM * DIM, CLASSES))
            x_train[sample_idx, :, :, :] = x_aug
            y_train[sample_idx, :, :] = y_aug
            sample_idx += 1
        self.train_idx = train_to
        return x_train, y_train

    def flow(self):
        x_train, y_train = self.flow_training()
        to_drop = len(x_train) % self.minibatch
        if to_drop > 0:
            x_train = x_train[:-to_drop, :, :, :]
            y_train = y_train[:-to_drop, :, :]
        return x_train, y_train

    def has_data(self):
        return self.train_idx < len(self.train_names)

    def reset(self):
        self.train_idx = 0
        shuffle(self.train_names)


def create_model():
    model = Sequential()
    # first convolutional block
    model.add(Convolution2D(32, 3, 3, input_shape=(DIM, DIM, CHANNELS),
                            border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # second convolutional block
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # third convolutional block
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # back to the original dim
    model.add(Convolution2D(1024, 5, 5,
                            border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 1, 1,
                            border_mode='same'))
    model.add(Deconvolution2D(CLASSES, 16, 16,
                              output_shape=(BATCHSIZE, DIM, DIM, CLASSES),
                              subsample=(8, 8),  border_mode='same'))
    model.add(Reshape((DIM * DIM, CLASSES)))
    model.add(Activation('softmax'))
    optmzr = adam(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=optmzr,
                  metrics=['accuracy'])
    print(model.summary())
    return model


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description='Process predicted image, creates geojson')
    arg_parser.add_argument('--indir', required=True,
                            help='Dir with input feats')
    arg_parser.add_argument('--targetdir', required=True,
                            help='Dir with target feats')
    args = arg_parser.parse_args()
    if not os.path.isdir(args.indir):
        raise RuntimeError('Cant open indir %s' % args.indir)
    if not os.path.isdir(args.targetdir):
        raise RuntimeError('Cant open target dir %s' % args.targetdir)
    return args


def main():
    args = parse_args()
    model = create_model()
    data_generator = DataGenerator(args.indir, args.targetdir)
    for e in range(EPOCHS):
        print('epoch %d' % e)
        if e % SAVE_MODEL_EVERY_NTH_EPOCH == 0:
            model.save('model_%d.bin' % e)
        while data_generator.has_data():
            x_train, y_train = data_generator.flow()
            model.fit(x_train, y_train, nb_epoch=1,
                      batch_size=BATCHSIZE)
        data_generator.reset()
    model.save('model_final.bin')


if __name__ == '__main__':
    main()
