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

    def get_model_fn(self):

        def model_fn(features, labels, mode, params):

            feature_embeddings = []
            item_embeddings = []
            feature_square_embeddings = []

            for feature in ['uid', 'gender', 'bal']:
                feature_emb = tf.compat.v1.feature_column.input_layer(features, params['feature_columns'][feature])
                feature_embeddings.append(feature_emb)

            dnn_user_input = tf.concat(feature_embeddings, axis=1, name='user')

            neg_item_dnn_input = tf.compat.v1.feature_column.input_layer(features,
                                                                         params['feature_columns']['item_neg'])

            pos_item_dnn_input = tf.compat.v1.feature_column.input_layer(features, params['feature_columns']['item'])

            for unit in params['hidden_units']:
                dnn_user_input = tf.compat.v1.layers.dense(dnn_user_input, units=unit, activation=tf.nn.relu)
                dnn_user_input = tf.compat.v1.layers.batch_normalization(dnn_user_input)
                dnn_user_input = tf.compat.v1.layers.dropout(dnn_user_input)
                dnn_user_input = tf.nn.l2_normalize(dnn_user_input)

                # 正负样本的item需要share 参数
                neg_item_dnn_input = tf.compat.v1.layers.dense(neg_item_dnn_input, units=unit, activation=tf.nn.relu)
                neg_item_dnn_input = tf.compat.v1.layers.batch_normalization(neg_item_dnn_input)
                neg_item_dnn_input = tf.compat.v1.layers.dropout(neg_item_dnn_input)
                neg_item_dnn_input = tf.nn.l2_normalize(neg_item_dnn_input)

                pos_item_dnn_input = tf.compat.v1.layers.dense(pos_item_dnn_input, units=unit, activation=tf.nn.relu)
                pos_item_dnn_input = tf.compat.v1.layers.batch_normalization(pos_item_dnn_input)
                pos_item_dnn_input = tf.compat.v1.layers.dropout(pos_item_dnn_input)
                pos_item_dnn_input = tf.nn.l2_normalize(pos_item_dnn_input)

            dot_1 = tf.reduce_sum(tf.multiply(dnn_user_input, pos_item_dnn_input), axis=1, keepdims=True) / float(
                params['temperature'])

            dot_2 = tf.reduce_sum(tf.multiply(dnn_user_input, neg_item_dnn_input), axis=1, keepdims=True) / float(
                params['temperature'])

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {"user_emb": dnn_user_input,
                               "item_embedding": pos_item_dnn_input}
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
