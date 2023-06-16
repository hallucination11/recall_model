import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, BatchNormalization, Input
import numpy as np
import glob

feature_description = {
    'application': tf.io.FixedLenFeature([], tf.int64, 0),
    'approval': tf.io.FixedLenFeature([], tf.int64, 0),
    'uid': tf.io.FixedLenFeature([], tf.int64, 0),
    'gender': tf.io.FixedLenFeature([], tf.int64, 0),
    'bal': tf.io.FixedLenFeature([], tf.int64, 0),
    'prod_list': tf.io.VarLenFeature(tf.int64),
    'item': tf.io.FixedLenFeature([], tf.int64, 0),
    'label': tf.io.FixedLenFeature([], tf.int64, 0),
    'brand_id': tf.io.FixedLenFeature([], tf.int64, 0),
    'prod_id': tf.io.FixedLenFeature([], tf.int64, 0),
    'age': tf.io.FixedLenFeature([], tf.int64, 0),
    'mobile_level': tf.io.FixedLenFeature([], tf.int64, 0),
    'item_price': tf.io.FixedLenFeature([], tf.float32, 0)
}


def parser(record):
    read_data = tf.io.parse_example(serialized=record, features=feature_description)
    label = read_data.pop('label')
    # label = read_data['click'] > 0
    # a =tf.compat.v1.string_split(tf.squeeze(tf.sparse.to_dense(read_data['view_7_vendor_id_list'])), ',')
    # read_data['id_list'] = tf.sparse.to_dense(a)
    # a = read_data.pop('view_7_vendor_id_list')
    # print(tf.sparse.to_dense(read_data['view_7_vendor_id_list']))
    return read_data, label


path = 'D:/work/My_model/tf_model/positive_dataset/dataset/part-r-1.tfrecords'
# train_file = []
# train_file.extend(sorted(glob.glob(path + '/part-r-*')))
ds = tf.data.TFRecordDataset(path)
ds = ds.repeat(1).batch(1).shuffle(2)
ds = ds.map(lambda x: parser(x))

for feature_dict, label in ds:
    for k, v in feature_dict.items():
        print(k)
        print(v)
    exit()