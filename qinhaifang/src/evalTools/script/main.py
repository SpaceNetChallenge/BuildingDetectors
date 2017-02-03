#!/usr/bin/env python
# encoding=gbk
"""
"""

import os
import os.path
import logging
import logging.config

import setting
from preprocess_space_net_data import PreprocessSpaceNetData


def setup():
    if not os.path.isdir(setting.LOG_DIR):
        try:
            os.mkdir(setting.LOG_DIR)
        except OSError as e:
            msg = "Mkdir {} error: e".format(setting.LOG_DIR, e)
            sys.stderr.write(msg)
            sys.exit(1)
    logging.config.fileConfig(setting.LOGGING_CONF_FILE)


def process():
    """docstring for process"""
    preprocess_space_net = PreprocessSpaceNetData()
    preprocess_space_net.process()
    #preprocess_space_net.test()


if __name__ == '__main__':
    setup()
    process()
