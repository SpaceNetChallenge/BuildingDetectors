# --------------------------------------------------------
# Multitask Network Cascade
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from datasets.pascal_voc_det import PascalVOCDet
from datasets.pascal_voc_seg import PascalVOCSeg

__sets = {
    'spaceNet_train': (lambda: PascalVOCSeg('train', '2012', 'spaceNet_data/spaceNetData/')),
    'spaceNet_val': (lambda: PascalVOCSeg('val', '2012', 'spaceNet_data/spaceNetData/')),
    'spaceNet_trainval': (lambda: PascalVOCDet('trainval', '2007')),
    'spaceNet_test': (lambda: PascalVOCDet('test', '2007'))
}


def get_imdb(name):
    """ Get an imdb (image database) by name.
    """
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()


def list_imdbs():
    return __sets.keys()
