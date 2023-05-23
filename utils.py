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


class DNN(Layer):
    """The Multi Layer Percetron
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with
        shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape
        ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.
      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.
        - **activation**: Activation function to use.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.
        - **use_bn**: bool. Whether use BatchNormalization before activation or not.
        - **output_activation**: Activation function to use in the last layer.If ``None``,it will be same as ``activation``.
        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 output_activation=None,
                 seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_activation = output_activation
        self.seed = seed

        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        # if len(self.hidden_units) == 0:
        #     raise ValueError("hidden_units is empty")
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(
                                            hidden_units[i], hidden_units[i + 1]),
                                        initializer=glorot_normal(
                                            seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [Dropout(self.dropout_rate, seed=self.seed + i) for i in
                               range(len(self.hidden_units))]

        self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.hidden_units))]

        if self.output_activation:
            self.activation_layers[-1] = activation_layer(self.output_activation)

        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(
                deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])

            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)
            try:
                fc = self.activation_layers[i](fc, training=training)
            except TypeError as e:  # TypeError: call() got an unexpected keyword argument 'training'
                print("make sure the activation function use training flag properly", e)
                fc = self.activation_layers[i](fc)

            fc = self.dropout_layers[i](fc, training=training)
            deep_input = fc

        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate,
                  'output_activation': self.output_activation, 'seed': self.seed}
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
