from unpack_data import get_mnist
import numpy as np
import tensorflow as tf
import tensorflow.keras.optimizers as optimizers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import progressbar
from generate_mask import adjacency_mask
import functools


class GaussianModel(tf.keras.layers.Layer):
    def __init__(self, shape, mask=None, mask_type="orthogonal", params=None):
        super(GaussianModel, self).__init__()
        if mask is None:
            self.mask = adjacency_mask(shape=(shape[0], shape[1]), mask_type=mask_type)
        else:
            self.mask = mask
        if params is None:
            par = Params(shape, self.mask)
        else:
            par = params

        self.loc = par.mu
        self.precision_matrix = par.precision_matrix

        prec_inv = tf.linalg.inv(self.precision_matrix)
        cov_det = tf.linalg.det(
            prec_inv
        )  # Apparently -inf. Det(precision)=0 according to tensorflow
        self.z = tf.Variable(1, dtype=tf.float32)
        """(2 * np.pi) ** (0.5 * dim) * tf.abs(
            cov_det
        ) ** 0.5"""

    def __call__(self, data):
        y_halfway = tf.einsum("ij,kj->ki", self.precision_matrix, data)
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

    def grad_log_model_old(self, data):
        grad_x = -tf.einsum("ij,sj->si", self.cov, (data - self.loc))
        return grad_x

    def grad_log_model(self, data):
        grad_x = -tf.einsum("ij,sj->si", self.precision_matrix, (data - self.loc))
        return grad_x

    def laplace_log_model_old(self, data):
        return -tf.einsum("ii", self.cov)

    def laplace_log_model(self, data):
        return -tf.einsum("ii", self.precision_matrix)

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

    def test_loss(self):
        return tf.reduce_sum(self.precision_matrix)+tf.reduce_sum(self.loc)

    def data_point_weights(
        self, data, det_noise_precision_matrix, noise_precision_matrix
    ):
        weights = tf.sqrt(
            det_noise_precision_matrix / tf.linalg.det(self.precision_matrix)
        ) * tf.exp(
            tf.einsum(
                "ki,ki->k",
                data,
                tf.einsum(
                    "ij,kj->ki", noise_precision_matrix - self.precision_matrix, data
                ),
            )
        )
        return weights

    def NCE(
        self,
        real_data,
        noise_data,
        noise_indicator_array,
        det_noise_precision_matrix,
        nosie_precision_matrix,
        nu,
        real_weights,
        noise_weights,
    ):
        M = tf.math.count_nonzero(noise_indicator_array)
        N = len(noise_indicator_array) - M
        term_1_and_4 = (
            1  # nu
            / M
            * (
                tf.einsum(
                    "ki,ki",
                    real_data,
                    tf.einsum(
                        "ij,kj->ki",
                        nosie_precision_matrix - self.precision_matrix,
                        real_data,
                    ),
                )
                # - tf.reduce_sum(tf.math.log(nu * real_weights + 1))
                - tf.reduce_sum(tf.math.log(1 * real_weights + 1))
            )
        )
        term_3 = -1 / N * tf.reduce_sum(tf.math.log(noise_weights + 1))

        term_2 = (
            -nu
            / 2
            * tf.math.log(
                nu
                ^ 2 * det_noise_precision_matrix / tf.linalg.det(self.precision_matrix)
            )
        )

        return term_1_and_4 + term_2 + term_3


class Params:
    def __init__(self, shape, mask):
        # Temporary values for testing:
        # self.cov=tf.Variable(tf.eye(dim),dtype=tf.float32) # This works
        rows = shape[0]
        cols = shape[1]
        dim = rows * cols
        self.mu = tf.zeros((1, dim), dtype=tf.float32)

        dim = shape[0] * shape[1]

        arr = tf.random.uniform((dim, dim), 0.1, 1, dtype=tf.float32) * mask
        self.diag_precision = tf.Variable(
            tf.linalg.band_part(arr, 0, 0), dtype=tf.float32
        )
        self.upper_precision = (
            tf.Variable(tf.linalg.band_part(arr, 0, -1), dtype=tf.float32)
            - self.diag_precision
        )
        self.lower_precision = tf.linalg.matrix_transpose(self.upper_precision)

        self.precision_matrix = tf.Variable(
            self.upper_precision + self.diag_precision + self.lower_precision,
            dtype=tf.float32,
        )

    def get_distribution(self):
        tfp.distributions.MultivariateNormalFullCovariance(
            loc=self.mu,
            covariance_matrix=self.covariance_matrix,
            validate_args=False,
            allow_nan_stats=True,
            name="MultivariateNormalFullCovariance",
        )

    def generate_samples(self, num):
        return np.random.multivariate_normal(
            mean=np.squeeze(self.mu.numpy()),
            cov=self.covariance_matrix,
            size=num,
            check_valid="ignore",
        ).astype(np.float32)

    @property
    def covariance_matrix(self):
        return tf.linalg.inv(self.precision_matrix)


learning_rate = 0.001
eta = 0.75  # Probability that sample is real and not noise
nu = 1 / eta - 1
epochs = 3
batch_size = 10
NCE = False

mask = adjacency_mask(shape=(28, 28), mask_type="orthogonal")
params = Params((28, 28), mask=mask)
model = GaussianModel((28, 28), mask=mask, params=params)

noise_params = Params((28, 28), mask=mask)
det_noise_precision_matrix = tf.linalg.det(noise_params.precision_matrix)
noise_precision_matrix = noise_params.precision_matrix

loss_fcn = model.model_loss
loss_fcn_NCE = model.NCE
optimizer = optimizers.RMSprop(learning_rate=learning_rate)


@tf.function
def train_step_nce(
    real_data,
    noise_data,
    noise_indicator_array,
    det_noise_precision_matrix,
    noise_precision_matrix,
    nu,
):
    with tf.GradientTape() as tape:
        noise_weights = model.data_point_weights(
            noise_data, det_noise_precision_matrix, noise_precision_matrix
        )
        real_weights = model.data_point_weights(
            real_data, det_noise_precision_matrix, noise_precision_matrix
        )
        loss = loss_fcn_NCE(
            real_data,
            noise_data,
            noise_indicator_array,
            det_noise_precision_matrix,
            noise_precision_matrix,
            nu,
            real_weights,
            noise_weights,
        )

    grads = tape.gradient(loss, model.precision_matrix) * model.mask

    grads_diag = tf.linalg.band_part(grads, 0, 0)
    grads_upper = tf.linalg.band_part(grads, 0, -1) - grads_diag
    grads_lower = tf.linalg.matrix_transpose(grads_upper)
    grads = grads_diag + grads_upper + grads_lower

    optimizer.apply_gradients(zip([grads], [model.precision_matrix]))
    return loss, grads


@tf.function
def train_step(data):
    with tf.GradientTape(persistent=True) as tape:

        loss = loss_fcn(
            data,
        )

    grads = tape.gradient(loss, model.precision_matrix) * model.mask
    grads_diag = tf.linalg.band_part(grads, 0, 0)
    grads_upper = tf.linalg.band_part(grads, 0, -1) - grads_diag
    grads_lower = tf.linalg.matrix_transpose(grads_upper)
    grads = grads_diag + grads_upper + grads_lower

    optimizer.apply_gradients(zip([grads], [model.precision_matrix]))

    loc_grads = tape.gradient(loss, model.loc)
    optimizer.apply_gradients(zip([loc_grads], [model.loc]))

    return loss, grads


data_dictionary = get_mnist()

train_images = data_dictionary["train-images"]
train_labels = data_dictionary["train-labels"]

train_images = train_images / 255

train_images += np.random.normal(scale=1 / 100, size=np.shape(train_images))

shape = np.shape(train_images)
train_images = np.reshape(train_images, (shape[0], shape[1] * shape[2]))

train_images = [img - np.mean(img) for img in train_images]
train_images = np.array(train_images, dtype=np.single)

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size=batch_size)


widgets = [
    " [",
    progressbar.Timer(format="Elapsed time: %(elapsed)s"),
    "] ",
    progressbar.Bar("*"),
    " (",
    progressbar.ETA(),
    ") ",
]

precision_matrix_before = model.precision_matrix.numpy()
loss = []
grads = []
for epoch in range(epochs):
    print("Start of epoch ", epoch + 1, "\n")
    bar = progressbar.ProgressBar(max_value=len(train_dataset), widgets=widgets).start()
    for step, (train_image_batch, _) in enumerate(train_dataset):
        if NCE:
            train_image_batch = tf.Variable(train_image_batch)
            noise_image_batch = tf.Variable(noise_params.generate_samples(batch_size))
            noise_indicator_array = np.random.binomial(1, eta, batch_size)
            tmp_real = []
            tmp_noise = []
            for index, indicator in enumerate(noise_indicator_array):
                if indicator == 1:
                    tmp_real.append(train_image_batch[index])
                else:
                    tmp_noise.append(noise_image_batch[index])

            """sort_index = np.argsort(noise_indicator_array)
            for ind in range(len(tmp)):
                train_image_batch[ind].assign(tmp[sort_index[ind]])
            noise_indicator_array = noise_indicator_array[sort_index]"""

            train_image_batch = tf.convert_to_tensor(tmp_real)
            noise_image_batch = tf.convert_to_tensor(tmp_noise)

            step_loss, step_grads = train_step_nce(
                train_image_batch,
                noise_image_batch,
                noise_indicator_array,
                det_noise_precision_matrix,
                noise_precision_matrix,
                nu,
            )
        else:
            step_loss, step_grads = train_step(train_image_batch)

        if step % 100 == 0:
            grads.append(step_grads.numpy())
            loss.append(step_loss)
        bar.update(step)

plt.plot(loss)
plt.show()

precision_matrix_after = model.precision_matrix.numpy()

plt.imshow(model.precision_matrix.numpy(), vmin=0, vmax=1)
plt.show()

plt.imshow(precision_matrix_before - precision_matrix_after)
plt.show()

np.save("gradients.npy", grads)
np.save("precision_matrix_before.npy", precision_matrix_before)
np.save("precision_matrix_after.npy", precision_matrix_after)

'''# Test of forcing symmetric gradients.
with tf.GradientTape(persistent=True) as tape:
    out2 = model(train_images[:3])
    print(out2)
    loss = model.test_loss()
    print(loss)
    loc = model.loc

print(model.loc)

grads = tape.gradient(loc, model.loc)
#grads = tape.gradient(loss, [model.precision_matrix, model.loc])

print(grads)
optimizer.apply_gradients(zip([grads], [model.precision_matrix]))'''
