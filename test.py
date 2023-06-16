import tensorflow as tf

a = tf.nn.softmax_cross_entropy_with_logits(
                labels=[[1, 0, 0], [1, 0, 0]], logits=[[1.2, 3.2, 4.2], [1.2, 3.2, 4.2]])
b = tf.matmul([[1, 2], [1, 2]], [[1, 2], [1, 2]], transpose_b=True)
print(b)