import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import scipy.linalg

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
        self.dense_layer_1 = tf.keras.layers.Dense(self.K1, activation='sigmoid', use_bias=False)  # s(Vx)
        self.dense_layer_2 = tf.keras.layers.Dense(self.K2, activation='none', use_bias=True)  # w^T x + c

        # Since we have expressions for both f_theta(x,z) and log p_theta(x),
        # we probably need to extract the weight matrices and biases at some point.

    @property
    def V(self):
        return self.dense_layer_1.get_weights()

    @property
    def W(self):
        return self.dense_layer_2.get_weights()

    @property
    def c(self):
        return self.dense_layer_2.get_bias()  # <-- Not 100% sure this is correct.

    def __call__(self, x):
        y = self.exponent_marginal_dist(x)
        return tf.exp(y)

    def build(self):
        self.built = True

    # log p_theta(x,z) = f_theta(x,y)
    def exponent_joint_dist(self, x, z):
        g = self.dense_layer_1(x)  # Compute g_theta(x)
        u = self.dense_layer_2(g)  # Compute W g_theta(x) + c
        y = tf.tensordot(u, z, axis=1)  # Compute z^T (W g_theta(x) + c)
        y += tf.tensordot(self.b, x, axis=1)  # Add b^T x
        y -= tf.tensordot(x, x, axis=1) / (2 * self.sigma ** 2)  # Subtract ||x||² / 2sigma²

        return y

    # log p_theta(x)
    def exponent_marginal_dist(self, x):
        g = self.dense_layer_1(x)  # Compute g_theta(x) = s(Vx)
        u = self.dense_layer_2(g)  # Compute w^T g_theta(x) + c

        y = self.S(u)  # Compute S(w^T g_theta(x) + c)
        y = tf.reduce_sum(y)  # tf.tensordot(y, tf.ones((1, dim), dtype=tf.float32), axis=1)  # sum_k S(w_k^T x + c_k)
        y += tf.tensordot(self.b, x, axis=1)  # Add b^T x
        y -= tf.tensordot(x, x, axis=1) / (2 * self.sigma ** 2)  # Subtract ||x||² / 2sigma²
        return y

    """
    Could rewrite S in terms of s or vice versa.
    S(u) = -log(s(-u)) or something like that.
    Might allow us to use activation=sigmoid in the second dense layer
    and never bother defining S and s in the first place.
    """
    def S(self, u):
        if u <= 0:
            return tf.math.log(1 + tf.exp(u))
        else:
            return u + tf.math.log(1 + tf.exp(u))

    def s(self, u):
        if u <= 0:
            return tf.exp(u) / (1 + tf.exp(u))
        else:
            return 1 / (1 + tf.exp(-u))

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
    @property
    def grad_log_model(self, data):
        grad_x = -tf.einsum("ij,sj->si", self.cov, (data - self.loc))
        return grad_x

    def laplace_log_model(self, data):
        return -tf.einsum("ii", self.cov)

    def model_loss(self, data):
        # 1/N sum i from 1 to N (1/2*||grad_x log model(xi)||^2 + laplace log model(xi))
        N = len(data)
        loss = 0
        for sample in range(N):
            loss = (
                    loss
                    + 1 / 2 * tf.linalg.norm(self.grad_log_model(data[sample])) ** 2
                    + self.laplace_log_model(data[sample])
            )

        loss = 1 / N * loss
        return loss


class Params:
    def __init__(self, shape, mask):
        rows = shape[0]
        cols = shape[1]
        dim = rows * cols

        """
        Parameters V, W, c are already part of the dense layers
        and should already be included in the trainable parameters.
        Does tf understand that trainable parameters = { dense layer params + b}?
        """

        vec = tf.random.normal(size=(1, dim))
        self.b = tf.Variable(
            vec,
            dtype=tf.float32,
            trainable=True,
        )

        self.sigma = 1  # 0.1 # Hyperparameter

def whiten(data):
    data = data - np.sum(data,axis=0)/len(data)
    c = 1/(len(data)-1)*np.transpose(data)@data
    eig_val,U = np.linalg.eig(c)
    Lambda = np.diag(eig_val)
    data_transpose = np.transpose(U)@scipy.linalg.sqrtm(np.linalg.inv(Lambda))@np.transpose(U)@np.transpose(data)
    return np.transpose(data_transpose)