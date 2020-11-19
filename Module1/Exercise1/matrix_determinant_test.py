import numpy as np
import tensorflow as tf

mat = tf.Variable(np.random.random((10, 10)))


@tf.function
def determinant(mat):
    return tf.linalg.det(mat)


with tf.GradientTape() as tape:
    det = 1/determinant(mat)

print(tape.gradient(det, mat))
