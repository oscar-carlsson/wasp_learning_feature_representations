import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import scipy.linalg
from tensorflow.keras.activations import sigmoid
import matplotlib.pyplot as plt

"""
 IMPORTANT: Just like the presentation, I have been sloppy
 regarding the use of w or W. Both these letters refer to
 the same matrix. (Presentation maybe uses W for the matrix
 and w_k for rows in W. But let's use consistent notation.)
"""


class DEM(tf.keras.layers.Layer):
    def __init__(self, shape, mask_type="orthogonal"):
        super(DEM, self).__init__()
        par = Params(shape)
        # Dense layers replace the parameters V, W, c
        self.b = par.b  # Bias vector
        self.K1 = par.K1  # Channels in dense layer 1
        self.K2 = par.K2  # Channels in dense layer 2
        self.sigma = par.sigma  # sigma in {1, 0.1}
        self.c = par.c
        self.V = par.V
        self.W = par.W
        # self.dense_layer_1 = tf.keras.layers.Dense(self.K1, activation='sigmoid', use_bias=False)  # s(Vx)
        # self.dense_layer_2 = tf.keras.layers.Dense(self.K2, activation=None, use_bias=True)  # w^T x + c

        # Since we have expressions for both f_theta(x,z) and log p_theta(x),
        # we probably need to extract the weight matrices and biases at some point.

    """@property
    def V(self):
        return self.dense_layer_1.get_weights()

    @property
    def W(self):
        return self.dense_layer_2.get_weights()

    @property
    def c(self):
        return self.dense_layer_2.get_bias()  # <-- Not 100% sure this is correct."""

    def __call__(self, x):
        y = self.exponent_marginal_dist(x)
        return tf.exp(y)

    def build(self):
        self.built = True

    @property
    def trainable_variables(self):
        return [self.b, self.c, self.V, self.W]

    def dense_layer_1(self, data):
        """
        Computes s(Vx)
        """
        # return sigmoid(tf.einsum("ij,kj->ki", self.V, data))
        return tf.map_fn(self.s, tf.einsum("ij,kj->ki", self.V, data))

    def dense_layer_2(self, data):
        return tf.einsum("ij,kj->ki", self.W, data) + self.c

    # log p_theta(x,z) = f_theta(x,y)
    def exponent_joint_dist(self, x, z):
        g = self.dense_layer_1(x)  # Compute g_theta(x)
        u = self.dense_layer_2(g)  # Compute W g_theta(x) + c
        y = tf.tensordot(u, z, axis=1)  # Compute z^T (W g_theta(x) + c)
        y += tf.tensordot(self.b, x, axis=1)  # Add b^T x
        y -= tf.tensordot(x, x, axis=1) / (
            2 * self.sigma ** 2
        )  # Subtract ||x||² / 2sigma²

        return y

    # log p_theta(x)
    def exponent_marginal_dist(self, x):
        g = self.dense_layer_1(x)  # Compute g_theta(x) = s(Vx)
        u = self.dense_layer_2(g)  # Compute w^T g_theta(x) + c

        y = tf.map_fn(self.S, u)  # Compute S(w^T g_theta(x) + c)
        y = tf.reshape(
            tf.reduce_sum(y, axis=1), (tf.shape(x)[0], 1)
        )  # tf.tensordot(y, tf.ones((1, dim), dtype=tf.float32), axis=1)  # sum_k S(w_k^T x + c_k)
        y += tf.einsum(
            "ij,kj->ki", self.b, x
        )  # tf.tensordot(self.b, x, axes=1)  # Add b^T x
        y -= tf.reshape(tf.einsum("ki,ki->k", x, x), (tf.shape(x)[0], 1)) / (
            2 * self.sigma ** 2
        )  # Subtract ||x||² / 2sigma², tf.tensordot(x, x, axes=1)
        return y

    """
    Could rewrite S in terms of s or vice versa.
    S(u) = -log(s(-u)) or something like that.
    Might allow us to use activation=sigmoid in the second dense layer
    and never bother defining S and s in the first place.
    """
    """def S(self, u):
        if u <= 0:
            return tf.math.log(1 + tf.exp(u))
        else:
            return u + tf.math.log(1 + tf.exp(u))"""

    def S(self, u):
        return tf.map_fn(self.S_aux, u)

    def S_aux(self, u):
        return tf.cond(
            u <= 0,
            lambda: tf.math.log(1 + tf.exp(u)),
            lambda: u + tf.math.log(1 + tf.exp(u)),
        )

    def S_prime(self, u):
        return self.s(u)

    def S_bis(self, u):
        return self.s_prime(u)

    def s(self, u):
        return tf.map_fn(self.s_aux, u)

    def s_aux(self, u):
        return tf.cond(
            u <= 0, lambda: tf.exp(u) / (1 + tf.exp(u)), lambda: 1 / (1 + tf.exp(-u))
        )

    def s_prime(self, u):
        return tf.map_fn(self.s_prime_aux, u)

    def s_prime_aux(self, u):
        return tf.cond(
            u <= 0,
            lambda: 1 / (1 + tf.exp(u)) ** 2,
            lambda: -tf.exp(-u) / (1 + tf.exp(-u)) ** 2,
        )

    def s_bis(self, u):
        return tf.map_fn(self.s_bis_aux, u)

    def s_bis_aux(self, u):
        return tf.cond(
            u <= 0,
            lambda: (tf.exp(u) - tf.exp(3 * u)) / (1 + tf.exp(u)) ** 4,
            lambda: (tf.exp(-3 * u) - tf.exp(-u)) / (1 + tf.exp(-u)) ** 4,
        )

    def grad_log_DEM(self, data):
        """
        If my calculations are correct the gradient of log(model) is:
        -1/sigma^2 * x_i + b_i + sum_{n=1}^{K}S'(w^T_n * s(Vx) + c_n) * w^T_n * s'(Vx) * V_{ni}
        I'm not sure about the last indices on V but I hope they're correct. The output shape seems to be fine.
        :param data:
        :return:
        """
        g = self.dense_layer_1(data)  # Compute g_theta(x) = s(Vx)
        u = self.dense_layer_2(g)  # Compute w^T g_theta(x) + c

        S_prime = tf.map_fn(self.S_prime, u)
        w_tn_s_prime = tf.einsum(
            "ij,kj->ki",
            self.W,
            tf.map_fn(self.s_prime, tf.einsum("ij,kj->ki", self.V, data)),
        )

        middle_step = S_prime * w_tn_s_prime
        sum = tf.einsum("kn,ni->ki", middle_step, self.V)  # Summing over n (small k in slides)

        return -1 / self.sigma ** 2 * data + self.b + sum

    def laplace_log_DEM(self, data):
        """
        I think my calcs are correct but they look like a mess...
        laplace(log(p))=
            -1/sigma^2
            + sum_{n=1}{K}(
                S''(w^T_n * s(Vx) + c_n) * (w^T_n * s'(Vx))^2 * sum_{i}V_ni^2
                + S'(w^T_n * s(Vx) + c_n) * (w^T_n * s''(Vx)) * sum_{i}V_ni^2
            )
        :param data:
        :return:
        """

        g = self.dense_layer_1(data)  # Compute g_theta(x) = s(Vx)
        u = self.dense_layer_2(g)  # Compute w^T g_theta(x) + c

        Vx = tf.einsum("ij,kj->ki", self.V, data)

        S_prime = tf.map_fn(self.S_prime, u)
        S_bis = tf.map_fn(self.S_bis, u)

        s_prime = tf.map_fn(self.s_prime, Vx)
        s_bis = tf.map_fn(self.s_bis, Vx)

        sum_i_V_ni2 = tf.reduce_sum(self.V,axis=1)

        w_tn_s_prime = tf.einsum(
            "ij,kj->ki",
            self.W,
            tf.map_fn(self.s_prime, tf.einsum("ij,kj->ki", self.V, data)),
        )

        w_tn_s_bis = tf.einsum(
            "ij,kj->ki",
            self.W,
            tf.map_fn(self.s_bis, tf.einsum("ij,kj->ki", self.V, data)),
        )

        term_1 = S_bis * w_tn_s_prime**2 * sum_i_V_ni2
        term_2 = S_prime * w_tn_s_bis * sum_i_V_ni2
        sum = tf.reduce_sum(term_1+term_2, axis=1)

        return -1/self.sigma**2 + sum

    # Latent variable
    def z(self, logits):
        """
        logits = Tensor representing the log-odds of a 1 event.
        Should therefore input logits = tf.log( s(w^T g(x) + c) ).
        Each entry is an independent Bernoulli distribution.
        """
        return tfp.distributions.Bernoulli(logits=logits)

    """
    grad_log_model, laplace_log_model and model_loss
    are untouched from Exercise 1. Haven't changed these yet.
    """

    def score_matching(self, data):
        # 1/N sum i from 1 to N (1/2*||grad_x log model(xi)||^2 + laplace log model(xi))
        N = len(data)
        '''loss = 0
        for sample in range(N):
            loss = (
                loss
                + 1 / 2 * tf.linalg.norm(self.grad_log_DEM(data[sample])) ** 2
                + self.laplace_log_DEM(data[sample])
            )

        loss = 1 / N * loss'''
        loss = 1/2 * tf.map_fn(tf.linalg.norm, self.grad_log_DEM(data)) ** 2 + self.laplace_log_DEM(data)
        loss = 1/N * tf.reduce_sum(loss)
        return loss


class Params:
    def __init__(self, shape):
        rows = shape[0]
        cols = shape[1]
        dim = rows * cols

        """
        Parameters V, W, c are already part of the dense layers
        and should already be included in the trainable parameters.
        Does tf understand that trainable parameters = { dense layer params + b}?
        """

        self.K1 = 64
        self.K2 = 64
        self.V = tf.Variable(
            tf.random.normal(shape=(self.K1, dim)), trainable=True, name="V"
        )
        self.W = tf.Variable(
            tf.random.normal(shape=(self.K2, self.K1)), trainable=True, name="W"
        )

        self.c = tf.Variable(
            tf.random.normal(shape=(1, self.K2)), trainable=True, name="c"
        )

        vec = tf.random.normal(shape=(1, dim))
        self.b = tf.Variable(
            vec,
            dtype=tf.float32,
            trainable=True,
            name="b",
        )

        self.sigma = 1  # 0.1 # Hyperparameter


def whiten(data):
    data = data - np.sum(data, axis=0) / len(data)
    c = 1 / (len(data) - 1) * np.transpose(data) @ data
    eig_val, U = np.linalg.eig(c)
    Lambda = np.diag(eig_val)
    data_transpose = (
        np.transpose(U)
        @ scipy.linalg.sqrtm(np.linalg.inv(Lambda))
        @ np.transpose(U)
        @ np.transpose(data)
    )
    return np.transpose(data_transpose)


def visualize_filters(V, margins=2, title=None, background_val=0):
    if len(np.shape(V)) == 2:
        num_filters = np.shape(V)[0]
        dim = np.shape(V)[1]
    elif len(np.shape(V)) == 1:
        num_filters = 1
        dim = len(V)
    else:
        raise ValueError("Only allow up to 2D filters.")

    if np.sqrt(dim).is_integer():
        filter_shape = (int(np.sqrt(dim)), int(np.sqrt(dim)))
    else:
        raise NotImplementedError("Not implemented for non-square images.")

    if num_filters != 1:
        filters = np.array([np.reshape(row, filter_shape) for row in V])
    else:
        filters = np.array([np.reshape(V, filter_shape)])

    if np.sqrt(num_filters).is_integer():
        num_rows = num_cols = int(np.sqrt(num_filters))
    else:
        raise NotImplementedError("Only implemented for n^2 number of filters.")

    canvas = np.full(
        (
            (num_rows + 1) * margins + num_rows * filter_shape[0],
            (num_cols + 1) * margins + num_cols * filter_shape[1],
        ),
        fill_value=background_val,
        dtype=np.single,
    )

    for ii in range(num_rows):
        for jj in range(num_cols):
            canvas[
                ii * filter_shape[0]
                + (ii + 1) * margins : (ii + 1) * filter_shape[0]
                + (ii + 1) * margins,
                jj * filter_shape[1]
                + (jj + 1) * margins : (jj + 1) * filter_shape[1]
                + (jj + 1) * margins,
            ] = filters[ii * num_cols + jj]

    plt.figure()
    plt.imshow(canvas)
    if isinstance(title, str):
        plt.title()
    plt.show()

    return canvas
