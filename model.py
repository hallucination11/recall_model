import collections

try:
    from tensorflow.python.ops.init_ops_v2 import Zeros, glorot_normal
except ImportError:
    from tensorflow.python.ops.init_ops import Zeros, glorot_normal_initializer as glorot_normal
from utils import *


class Model(collections.namedtuple("Model", ["model_name",
                                             'model_dir', 'embedding_upload_hook', 'high_param'])):
    def __new__(cls, model_name, model_dir, embedding_upload_hook=None, high_param=None):
        return super(Model, cls).__new__(cls, model_name, model_dir, embedding_upload_hook, high_param
                                         )

    def get_model_fn(self):
        def model_fn(features, labels, mode, params):
            pass

        return model_fn

    def get_estimator(self):
        estimator = tf.estimator.Estimator(model_dir=self.model_dir, model_fn=self.get_model_fn(), params={})
        # add gauc

        return estimator


# recall
class DSSM(Model):

    def get_model_fn(self):

        def model_fn(features, labels, mode, params):

            # sampler_config
            # train_inbatch_counter = Counter(features[item_name])
            # mast be a tensor
            # train_inbatch_counter = Counter(item_list)
            feature_embeddings = []
            item_embeddings = []
            feature_square_embeddings = []

            for feature in ['uid', 'item', 'gender', 'bal']:
                feature_emb = tf.compat.v1.feature_column.input_layer(features, params['feature_columns'][feature])
                if feature != 'item':
                    feature_embeddings.append(feature_emb)
                    feature_square_embeddings.append(tf.square(feature_emb))
                else:
                    item_embeddings.append(feature_emb)

            user_net = tf.concat(feature_embeddings, axis=1, name='user')
            item_net = tf.concat(item_embeddings, axis=1, name='item')

            for unit in params['hidden_units']:
                user_net = tf.compat.v1.layers.dense(user_net, units=unit, activation=tf.nn.relu)
                user_net = tf.compat.v1.layers.batch_normalization(user_net)
                user_net = tf.compat.v1.layers.dropout(user_net)
                user_net = tf.nn.l2_normalize(user_net)

                item_net = tf.compat.v1.layers.dense(item_net, units=unit, activation=tf.nn.relu)
                item_net = tf.compat.v1.layers.batch_normalization(item_net)
                item_net = tf.compat.v1.layers.dropout(item_net)
                item_net = tf.nn.l2_normalize(item_net)

            if self.high_param['loss_type'] == 'sigmoid':
                dot = tf.reduce_sum(tf.multiply(user_net, item_net), axis=1, keepdims=True) / float(
                    params['temperature'])
                logits = tf.sigmoid(dot)
            if self.high_param['loss_type'] == 'softmax':
                return None

            if mode == tf.estimator.ModeKeys.PREDICT:
                approval_pred = tf.argmax(logits, axis=-1)
                predictions = {"approval_pred": approval_pred,
                               "user_emb": user_net,
                               "probability": logits}
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

            approval_label = tf.reshape(labels['approval'], shape=[-1, 1])
            loss = tf.compat.v1.losses.log_loss(approval_label, logits)

            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
                train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

            if mode == tf.estimator.ModeKeys.EVAL:
                approval_pred_label = tf.argmax(logits, axis=-1)
                approval_auc = tf.compat.v1.metrics.auc(labels=approval_label, predictions=approval_pred_label)
                return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'approval_auc': approval_auc})

        # self.embedding_upload_hook.id_list_embedding = id_list_embedding

        return model_fn

    def get_estimator(self):
        # 商品id类特征
        def get_categorical_hash_bucket_column(key, hash_bucket_size, dimension, dtype):
            categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
                key, hash_bucket_size=hash_bucket_size, dtype=dtype)
            return tf.feature_column.embedding_column(categorical_column, dimension=dimension)

        # 连续值类特征（差异较为明显）
        def get_bucketized_column(key, boundaries, dimension):
            bucketized_column = tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key), boundaries)
            return tf.feature_column.embedding_column(bucketized_column, dimension=dimension)

        long_id_feature_columns = {}

        cnt_feature_columns = {
            "uid": get_categorical_hash_bucket_column("uid", hash_bucket_size=2000, dimension=6, dtype=tf.int64),
            "item": get_categorical_hash_bucket_column("item", hash_bucket_size=100, dimension=3, dtype=tf.int64),
            "bal": get_bucketized_column("bal", boundaries=[10002.0, 14158.35, 18489.0, 23177.0, 27839.8, 32521.5,
                                                            36666.7, 41386.9, 45919.6, 50264.55, 54345.0], dimension=4),
            "gender": get_categorical_hash_bucket_column("gender", hash_bucket_size=2, dimension=1, dtype=tf.int64)
        }

        all_feature_column = {}
        all_feature_column.update(long_id_feature_columns)
        all_feature_column.update(cnt_feature_columns)

        weight_column = tf.feature_column.numeric_column('weight')

        hidden_layers = [256, 128]

        num_experts = 3

        task_names = ("ctr", "ctcvr", "ctvoi")

        task_types = ("binary", "binary", "binary")

        lamda = 1

        estimator = tf.estimator.Estimator(
            model_dir=self.model_dir,
            model_fn=self.get_model_fn(),
            params={
                'hidden_units': hidden_layers,
                'feature_columns': all_feature_column,
                'weight_column': weight_column,
                'lamda': lamda,
                'num_experts': num_experts,
                'task_names': task_names,
                'task_types': task_types,
                'gate_dnn_hidden_units': [10],
                'tower_dnn_hidden_units': 64,
                'temperature': self.high_param['temperature']
            })

        return estimator


class DSSM_Two_Pair_Loss(Model):

    def get_model_fn(self):

        def model_fn(features, labels, mode, params):

            feature_embeddings = []
            item_embeddings = []
            feature_square_embeddings = []

            # item and ite_neg share embedding
            pid_vocab_size = 100
            item_neg = tf.compat.v1.string_to_hash_bucket_fast(tf.as_string(features["item_neg"]), pid_vocab_size)
            item = tf.compat.v1.string_to_hash_bucket_fast(tf.as_string(features["item"]), pid_vocab_size)

            embedding_size = 16
            embeddings = tf.compat.v1.get_variable(name="embeddings", dtype=tf.float32,
                                                   shape=[pid_vocab_size, embedding_size])

            item_neg_emb = tf.nn.embedding_lookup(embeddings, item_neg)
            item_emb = tf.nn.embedding_lookup(embeddings, item)

            dnn_layer = DNN(hidden_units=params['hidden_units'], activation='relu', use_bn=True)

            item_output = dnn_layer(item_emb)
            item_neg_output = dnn_layer(item_neg_emb)

            for feature in ['uid', 'gender', 'bal']:
                feature_emb = tf.compat.v1.feature_column.input_layer(features, params['feature_columns'][feature])
                feature_embeddings.append(feature_emb)

            dnn_user_input = tf.concat(feature_embeddings, axis=1, name='user')

            for unit in params['hidden_units']:
                dnn_user_input = tf.compat.v1.layers.dense(dnn_user_input, units=unit, activation=tf.nn.relu)
                dnn_user_input = tf.compat.v1.layers.batch_normalization(dnn_user_input)
                dnn_user_input = tf.compat.v1.layers.dropout(dnn_user_input)
                dnn_user_input = tf.nn.l2_normalize(dnn_user_input)

            dot_1 = tf.reduce_sum(tf.multiply(dnn_user_input, item_output), axis=1, keepdims=True) / float(
                params['temperature'])

            dot_2 = tf.reduce_sum(tf.multiply(dnn_user_input, item_neg_output), axis=1, keepdims=True) / float(
                params['temperature'])

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {"user_emb": dnn_user_input,
                               "item_embedding": item_output}
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

            loss = tf.maximum(0, dot_1 - dot_2 + params['margin'])

            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
                train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

        # self.embedding_upload_hook.id_list_embedding = id_list_embedding

        return model_fn

    def get_estimator(self):
        # 商品id类特征
        def get_categorical_hash_bucket_column(key, hash_bucket_size, dimension, dtype):
            categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
                key, hash_bucket_size=hash_bucket_size, dtype=dtype)
            return tf.feature_column.embedding_column(categorical_column, dimension=dimension)

        # 连续值类特征（差异较为明显）
        def get_bucketized_column(key, boundaries, dimension):
            bucketized_column = tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key), boundaries)
            return tf.feature_column.embedding_column(bucketized_column, dimension=dimension)

        long_id_feature_columns = {}

        cnt_feature_columns = {
            "uid": get_categorical_hash_bucket_column("uid", hash_bucket_size=2000, dimension=6, dtype=tf.int64),
            "item": get_categorical_hash_bucket_column("item", hash_bucket_size=100, dimension=3, dtype=tf.int64),
            "bal": get_bucketized_column("bal", boundaries=[10002.0, 14158.35, 18489.0, 23177.0, 27839.8, 32521.5,
                                                            36666.7, 41386.9, 45919.6, 50264.55, 54345.0], dimension=4),
            "gender": get_categorical_hash_bucket_column("gender", hash_bucket_size=2, dimension=1, dtype=tf.int64),
            "item_neg": get_categorical_hash_bucket_column("item_neg", hash_bucket_size=100, dimension=3,
                                                           dtype=tf.int64)
        }

        all_feature_column = {}
        all_feature_column.update(long_id_feature_columns)
        all_feature_column.update(cnt_feature_columns)

        weight_column = tf.feature_column.numeric_column('weight')

        hidden_layers = [256, 128]

        num_experts = 3

        task_names = ("ctr", "ctcvr", "ctvoi")

        task_types = ("binary", "binary", "binary")

        lamda = 1

        estimator = tf.estimator.Estimator(
            model_dir=self.model_dir,
            model_fn=self.get_model_fn(),
            params={
                'hidden_units': hidden_layers,
                'feature_columns': all_feature_column,
                'weight_column': weight_column,
                'lamda': lamda,
                'num_experts': num_experts,
                'task_names': task_names,
                'task_types': task_types,
                'gate_dnn_hidden_units': [10],
                'tower_dnn_hidden_units': 64,
                'temperature': self.high_param['temperature'],
                'margin': self.high_param['margin']
            })

        return estimator


class DSSM_resnet(Model):

    def get_model_fn(self):

        def model_fn(features, labels, mode, params):

            # sampler_config
            # train_inbatch_counter = Counter(features[item_name])
            # mast be a tensor
            # train_inbatch_counter = Counter(item_list)
            feature_embeddings = []
            item_embeddings = []
            feature_square_embeddings = []

            for feature in ['uid', 'item', 'gender', 'bal']:
                feature_emb = tf.compat.v1.feature_column.input_layer(features, params['feature_columns'][feature])
                if feature != 'item':
                    feature_embeddings.append(feature_emb)
                    feature_square_embeddings.append(tf.square(feature_emb))
                else:
                    item_embeddings.append(feature_emb)

            user_net = tf.concat(feature_embeddings, axis=1, name='user')

            user_net_1 = tf.compat.v1.layers.dense(user_net, units=256, activation=tf.nn.relu)
            user_net_1 = tf.compat.v1.layers.batch_normalization(user_net_1)
            user_net_1 = tf.compat.v1.layers.dropout(user_net_1)
            user_net_1 = tf.nn.l2_normalize(user_net_1)

            user_net_2 = tf.compat.v1.layers.dense(user_net_1, units=128, activation=tf.nn.relu)
            user_net_2 = tf.compat.v1.layers.batch_normalization(user_net_2)
            user_net_2 = tf.compat.v1.layers.dropout(user_net_2)
            user_net_2 = tf.nn.l2_normalize(user_net_2)

            user_net_3 = tf.compat.v1.layers.dense(user_net_2, units=128, activation=tf.nn.relu)
            user_net_3 = tf.compat.v1.layers.batch_normalization(user_net_3)
            user_net_3 = tf.compat.v1.layers.dropout(user_net_3)
            user_net_3 = tf.nn.l2_normalize(user_net_3)

            user_net_concat = tf.concat([user_net_1, user_net_2, user_net_3], axis=1)

            item_net = tf.concat(item_embeddings, axis=1, name='item')

            item_net_1 = tf.compat.v1.layers.dense(item_net, units=256, activation=tf.nn.relu)
            item_net_1 = tf.compat.v1.layers.batch_normalization(item_net_1)
            item_net_1 = tf.compat.v1.layers.dropout(item_net_1)
            item_net_1 = tf.nn.l2_normalize(item_net_1)

            item_net_2 = tf.compat.v1.layers.dense(item_net_1, units=128, activation=tf.nn.relu)
            item_net_2 = tf.compat.v1.layers.batch_normalization(item_net_2)
            item_net_2 = tf.compat.v1.layers.dropout(item_net_2)
            item_net_2 = tf.nn.l2_normalize(item_net_2)

            item_net_3 = tf.compat.v1.layers.dense(item_net_2, units=64, activation=tf.nn.relu)
            item_net_3 = tf.compat.v1.layers.batch_normalization(item_net_3)
            item_net_3 = tf.compat.v1.layers.dropout(item_net_3)
            item_net_3 = tf.nn.l2_normalize(item_net_3)

            item_net_concat = tf.concat([item_net_1, item_net_2, item_net_3], axis=1)

            user_net_output = tf.compat.v1.layers.dense(user_net_concat, units=128, name='user_net')
            item_net_output = tf.compat.v1.layers.dense(item_net_concat, units=128, name='item_net')

            dot = tf.reduce_sum(tf.multiply(user_net_output, item_net_output), axis=1, keepdims=True) / float(
                params['temperature'])

            logits = tf.sigmoid(dot)

            if mode == tf.estimator.ModeKeys.PREDICT:
                approval_pred = tf.argmax(logits, axis=-1)
                predictions = {"approval_pred": approval_pred,
                               "user_emb": user_net,
                               "probability": logits}
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

            approval_label = tf.reshape(labels['approval'], shape=[-1, 1])
            loss = tf.compat.v1.losses.log_loss(approval_label, logits)

            if mode == tf.estimator.ModeKeys.TRAIN:
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
                train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

            if mode == tf.estimator.ModeKeys.EVAL:
                approval_pred_label = tf.argmax(logits, axis=-1)
                approval_auc = tf.compat.v1.metrics.auc(labels=approval_label, predictions=approval_pred_label)
                return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'approval_auc': approval_auc})

        # self.embedding_upload_hook.id_list_embedding = id_list_embedding

        return model_fn


    def get_estimator(self):
        # 商品id类特征
        def get_categorical_hash_bucket_column(key, hash_bucket_size, dimension, dtype):
            categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
                key, hash_bucket_size=hash_bucket_size, dtype=dtype)
            return tf.feature_column.embedding_column(categorical_column, dimension=dimension)

        # 连续值类特征（差异较为明显）
        def get_bucketized_column(key, boundaries, dimension):
            bucketized_column = tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key), boundaries)
            return tf.feature_column.embedding_column(bucketized_column, dimension=dimension)

        long_id_feature_columns = {}

        cnt_feature_columns = {
            "uid": get_categorical_hash_bucket_column("uid", hash_bucket_size=2000, dimension=6, dtype=tf.int64),
            "item": get_categorical_hash_bucket_column("item", hash_bucket_size=100, dimension=3, dtype=tf.int64),
            "bal": get_bucketized_column("bal", boundaries=[10002.0, 14158.35, 18489.0, 23177.0, 27839.8, 32521.5,
                                                            36666.7, 41386.9, 45919.6, 50264.55, 54345.0], dimension=4),
            "gender": get_categorical_hash_bucket_column("gender", hash_bucket_size=2, dimension=1, dtype=tf.int64)
        }

        all_feature_column = {}
        all_feature_column.update(long_id_feature_columns)
        all_feature_column.update(cnt_feature_columns)

        weight_column = tf.feature_column.numeric_column('weight')

        hidden_layers = [256, 128]

        num_experts = 3

        task_names = ("ctr", "ctcvr", "ctvoi")

        task_types = ("binary", "binary", "binary")

        lamda = 1

        estimator = tf.estimator.Estimator(
            model_dir=self.model_dir,
            model_fn=self.get_model_fn(),
            params={
                'hidden_units': hidden_layers,
                'feature_columns': all_feature_column,
                'weight_column': weight_column,
                'lamda': lamda,
                'gate_dnn_hidden_units': [10],
                'tower_dnn_hidden_units': 64,
                'temperature': self.high_param['temperature']
            })

        return estimator


