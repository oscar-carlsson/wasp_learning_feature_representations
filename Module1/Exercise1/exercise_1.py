from unpack_data import get_mnist
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

class GaussianModel(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(GaussianModel, self).__init__()
        par = Params(dim)
        self.loc = par.mu
        self.covariance_matrix = par.cov
        self.z = tf.Variable(1, dtype=tf.float32)
        """(2 * np.pi) ** (0.5 * dim) * tf.abs(
            tf.linalg.det(self.covariance_matrix)
        ) ** 0.5"""

    def __call__(self, data):
        y_halfway = tf.einsum("ij,kj->ki", self.covariance_matrix, data)
        y = tf.einsum("ki,ki->k", data, y_halfway)
        return tf.exp(-0.5 * y) / self.z

    def build(self):
        self.built = True

    """def get_multivariate_distrubution(self):
        distr = tfp.distributions.MultivariateNormalFullCovariance(
            loc=self.loc,
            covariance_matrix=self.cov,
            validate_args=False,
            allow_nan_stats=True,
            name="MultivariateNormalFullCovariance",
        )
        return distr"""

    @property
    def cov(self):
        return self.covariance_matrix

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
    def __init__(self, dim):
        # Temporary values for testing:
        # self.cov=tf.Variable(tf.eye(dim),dtype=tf.float32) # This works
        self.mu = tf.zeros((1, dim), dtype=tf.float32)
        self.cov = tf.Variable(
            np.random.random((dim, dim)), dtype=tf.float32, trainable=True
        )


data_dictionary = get_mnist()

train_images = data_dictionary["train-images"]

train_images = train_images / 255

train_images += np.random.normal(scale=1 / 100, size=np.shape(train_images))

shape = np.shape(train_images)
print(shape)

train_images = np.reshape(train_images, (shape[0], shape[1] * shape[2]))

train_images = [img - np.mean(img) for img in train_images]

train_images = tf.constant(train_images, dtype=tf.float32)

model = GaussianModel(28 * 28)
print(
    "Own layer output: ",
    model(tf.constant(train_images[:3], dtype=tf.float32)).numpy(),
)

with tf.GradientTape(persistent=True) as tape:
    out2 = model(train_images[:3])
    loss = model.model_loss(train_images[:3])

print(tape.gradient(loss, model.covariance_matrix))

print(model.model_loss(tf.constant(train_images[:3], dtype=tf.float32)))


"""distribution = model2.get_multivariate_distrubution()

sample = distribution.sample()

print(np.shape(sample))"""
