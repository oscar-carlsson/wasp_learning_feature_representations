import numpy as np
import tensorflow as tf
import sys
from generate_mask import adjacency_mask
import matplotlib.pyplot as plt

def check_inv(x):
    if np.linalg.cond(x) < 1 / sys.float_info.epsilon:
        print("Invertable!")
    else:
        print("Not invertable!")


def determinant_tf(mat):
    return tf.linalg.det(mat)


def determinant(mat):
    sign, logdet = np.linalg.slogdet(mat)
    return sign * np.exp(logdet)


class toy_model:
    def __init__(self, shape, mask):
        self.A = tf.Variable(np.random.random(shape))
        self.mask = mask

    @property
    def mat(self):
        tmp = self.A # tf.linalg.band_part(self.A, 0, -1)
        return tf.matmul(tf.linalg.matrix_transpose(tmp), tmp) * self.mask


shape = (10, 10)
dim = np.prod(shape)
mask = adjacency_mask(shape=shape, mask_type="orthogonal")
check_inv(mask)
mat = tf.Variable(np.random.random((dim, dim)) * mask)
noise_mat = tf.Variable(np.random.random((dim, dim)) * mask)
noise_mat_det = tf.linalg.det(noise_mat)

opt = tf.optimizers.Adam()

data = tf.Variable(np.random.random((100, dim)))
model = toy_model((dim, dim), mask)

with tf.GradientTape(persistent=True) as tape:
    det = tf.reduce_sum(model.mat)

    model_mat = model.mat
    model_mat_sum = tf.reduce_sum(model_mat)

    mat = tf.matmul(tf.linalg.matrix_transpose(model.A), model.A) * model.mask
    mat_sum = tf.reduce_sum(mat)

    out_model = tf.exp(
        -1 / 2 * tf.einsum("ki,ki->k", data, tf.einsum("ij,kj->ki", model.mat, data))
    )
    loss_model = tf.reduce_sum(out_model)

    out = tf.exp(
        -1 / 2 * tf.einsum("ki,ki->k", data, tf.einsum("ij,kj->ki", mat, data))
    )
    loss = tf.reduce_sum(out)

print(loss)
print('Gradient of sum of matrix elements extracted from the model:\n ',tape.gradient(model_mat_sum, model.A))
print('Gradient of sum of matrix elements constructed in the tape:\n ',tape.gradient(mat_sum,model.A))
print('Gradient of loss using matrix constructed in tape:\n ',tape.gradient(loss, model.A))
print('Gradient of loss using matrix constructed in model:\n ',tape.gradient(loss_model, model.A))
grad = tape.gradient(loss, model.A)
pre_grad_A = model.A
opt.apply_gradients(zip([grad], [model.A]))
post_grad_A = model.A

