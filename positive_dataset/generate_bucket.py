import tensorflow as tf
import numpy as np
import math

filename = 'D:/work/My_model/tf_model/positive_dataset/dataset/part-r-1.tfrecords'

feature_description = {
    'brand_id': tf.io.FixedLenFeature([], tf.int64, 0),
    'prod_id': tf.io.FixedLenFeature([], tf.int64, 0),
    'age': tf.io.FixedLenFeature([], tf.int64, 0),
    'mobile_level': tf.io.FixedLenFeature([], tf.int64, 0),
    'item_price': tf.io.FixedLenFeature([], tf.float32, 0),

}


def float_list_to_str(L):
    res = []
    for v in L:
        if v == int(v):
            res.append('{:.1f}'.format(v))
        else:
            res.append('{:.4f}'.format(v))
    return '[' + ', '.join(res) + ']'


def double_list_to_str(L):
    res = []
    for v in L:
        res.append(str(v))
    return '[' + ', '.join(res) + ']'


def parser(record):
    read_data = tf.io.parse_example(serialized=record, features=feature_description)
    return read_data


def print_features_v2(fv, features):
    for f in features:
        key = f  # strip off uid1_ and uid2_ prefix
        boundary = np.percentile(fv[key], range(0, 100, 5))
        boundary = sorted(list(set([round(x, 4) for x in boundary])))
        print(boundary)
        print("""
                embedding_columns([
                    bucketized_feature('{0}', {1}),
                ], {2}) +\\""".format(key, float_list_to_str(boundary), int(math.ceil(len(boundary) ** 0.25)) + 1),
              end="")


feature = ['brand_id', 'prod_id', 'age', 'mobile_level', 'item_price']


ds = tf.data.TFRecordDataset(filename, num_parallel_reads=1024)
ds = ds.batch(10000).map(parser)

fv = {f: np.array([]) for f in set([x for x in feature])}

for feature_dict in ds:
    for k, v in feature_dict.items():
        fv[k] = np.concatenate((fv[k], v.numpy()))

print_features_v2(fv, feature)
