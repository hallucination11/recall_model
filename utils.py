from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import datetime as dt
import logging
import os
import glob
from collections import namedtuple
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer, Activation
import itertools
from typing import List

try:
    from tensorflow.python.ops.init_ops_v2 import Zeros, Ones, glorot_normal, TruncatedNormal
except ImportError:
    from tensorflow.python.ops.init_ops import Zeros, Ones, glorot_normal_initializer as glorot_normal

from tensorflow.python.keras.layers import Layer, Dropout, Lambda, Add

try:
    from tensorflow.python.keras.layers import BatchNormalization
except ImportError:
    BatchNormalization = tf.keras.layers.BatchNormalization
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.ops import rnn_cell_impl

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"
# pylint: disable=protected-access
_concat = rnn_cell_impl._concat
_like_rnncell = rnn_cell_impl.assert_like_rnncell

logger = logging.getLogger(__name__)
feature_description = []


def parse_features(record):
    read_data = tf.io.parse_example(serialized=record,
                                    features=feature_description)
    click_rate = read_data.pop('click')  ##tfrecord中有对应点击率，pop出来作为label的判断依据

    label = click_rate > 0

    read_data['weight'] = tf.fill(tf.shape(label), 1.0)

    return read_data, label


def get_input_fn(filename, batch_size=1, compression="GZIP", n_repeat=1):
    def input_fn():
        ds = tf.data.TFRecordDataset(filename, compression)  ##压缩方式可选
        ds = ds.repeat(n_repeat).batch(batch_size)
        ds = ds.map(lambda x: parse_features(x))
        ds = ds.prefetch(buffer_size=batch_size)
        return ds

    return input_fn()


def get_days_between(start_date, end_date):
    '''
    :param start_date: str YYYY-MM-DD
    :param end_date: str YYYY-MM-DD
    :return:
    '''
    start_date = dt.date(*[int(x) for x in start_date.split('-')])
    end_date = dt.date(*[int(x) for x in end_date.split('-')])
    n_days = (end_date - start_date).days + 1
    assert (n_days > 0)
    return [str(start_date + dt.timedelta(x)) for x in range(n_days)]


def get_training_files(dirs, progress_filename="", resume=False):
    '''
    :param dirs:
    :param progress_filename:
    :param resume: 是否从中断处接着训练
    :return:
    '''
    files = []
    for directory in dirs:
        files.extend(sorted(glob.glob(directory + "/guess-r-*")))
    if resume:
        logger.info("Resume: {}".format(resume))
        if not os.path.exists(progress_filename):
            logger.warning("progress file '{}' doesn't exist".format(progress_filename))
            return files
        with open(progress_filename, 'r') as f:
            last_file_trained = f.read().strip()
            logger.info("last_file_trained: {}".format(last_file_trained))
        try:
            idx = files.index(last_file_trained)
            logger.info("last trained file {} is at position {} in the entire file list".format(last_file_trained, idx))
        except ValueError as e:
            logger.warning("last_file_trained '{}' not found in files. Got ValueError: {}. Returning all files.".format(
                last_file_trained, e))
            return files
        logger.info("return files from position {}".format(idx + 1))
        return files[idx + 1:]
    logger.info("return all files")
    return files


def batch_train_files(train_files, batch_size):
    assert batch_size > 0
    res = []
    for i in range(0, len(train_files), batch_size):
        res.append(train_files[i:i + batch_size])
    return res


def export_model(model, saved_model_dir, feature_spec):
    export_path = model.export_saved_model(saved_model_dir,
                                           tf.estimator.export.build_raw_serving_input_receiver_fn(
                                               feature_spec=feature_spec))
    return export_path





