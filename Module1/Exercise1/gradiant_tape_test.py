import numpy as np
import tensorflow as tf

mat = tf.Variable(np.random.random((10, 10)))
noise_mat = tf.Variable(np.random.random((10, 10)))
noise_mat_det = tf.linalg.det(noise_mat)


@tf.function
def determinant(mat):
    return tf.linalg.det(mat)


@tf.function
def weights(data, det, mat, noise_mat):
    tmp = tf.sqrt(det / tf.linalg.det(mat)) * tf.exp(
        tf.einsum(
            "ki,ki->k",
            data,
            tf.einsum("ij,kj->ki", noise_mat - mat, data),
        )
    )
    return tmp


data = tf.Variable(np.random.random((3, 10)))

with tf.GradientTape() as tape:
    w = weights(data, noise_mat_det, mat, noise_mat)
    det = determinant(mat)

print(tape.gradient(w, mat))
