from unpack_data import get_mnist
import numpy as np
import tensorflow as tf
import tensorflow.keras.optimizers as optimizers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import progressbar
from generate_mask import adjacency_mask
import functools
import time
import scipy.linalg

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

        self.dim = np.prod(shape)

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
    def covariance_matrix(self):
        return tf.linalg.inv(self.precision_matrix)

    """@property
    def precision_matrix(self):
        #return tf.matmul(tf.linalg.matrix_transpose(self.A), self.A) * self.mask
        return self.A + tf.linalg.matrix_transpose(self.A) - tf.linalg.band_part(self.A, 0, 0)"""

    @property
    def z(self):
        return (2 * np.pi) ** (0.5 * self.dim) * tf.abs(
            tf.linalg.det(self.covariance_matrix)
        ) ** 0.5

    def grad_log_model(self, data):
        grad_x = -tf.einsum("ij,sj->si", self.precision_matrix, (data - self.loc))
        return grad_x

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

    def data_point_weights(
        self, data, det_noise_precision_matrix, noise_precision_matrix
    ):
        weights = tf.sqrt(
            tf.math.abs(
                det_noise_precision_matrix / tf.linalg.det(self.precision_matrix)
            )
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

    def NCE_real_data_terms(
        self, real_data, det_noise_precision_matrix, noise_precision_matrix
    ):
        real_weights = model.data_point_weights(
            real_data, det_noise_precision_matrix, noise_precision_matrix
        )
        N = tf.shape(real_data).numpy()[0]
        term_3 = -1 / N * tf.reduce_sum(tf.math.log(real_weights + 1))
        return term_3

    def NCE_noise_data_terms(
        self, noise_data, nu, det_noise_precision_matrix, noise_precision_matrix
    ):
        M = tf.shape(noise_data).numpy()[0]
        noise_weights = self.data_point_weights(
            noise_data, det_noise_precision_matrix, noise_precision_matrix
        )
        term_1 = (
            nu
            / M
            * (
                tf.einsum(
                    "ki,ki",
                    noise_data,
                    tf.einsum(
                        "ij,kj->ki",
                        noise_precision_matrix - self.precision_matrix,
                        noise_data,
                    ),
                )
            )
        )
        term_4 = -nu / M * tf.reduce_sum(tf.math.log(nu * noise_weights + 1))
        return term_1 + term_4

    def NCE_precision_matrix_term(self, nu, det_noise_precision_matrix):
        term_2 = (
            -nu
            / 2
            * tf.math.log(
                nu ** 2
                * tf.math.abs(
                    det_noise_precision_matrix / tf.linalg.det(self.precision_matrix)
                )
                + np.finfo(float).eps
            )
        )
        return term_2

    def NCE_zero(self):
        return 0

    def NCE(
        self,
        real_data,
        noise_data,
        det_noise_precision_matrix,
        noise_precision_matrix,
        nu,
    ):
        loss = 0
        loss = loss + tf.cond(
            tf.shape(real_data).numpy()[0] != 0,
            lambda: self.NCE_real_data_terms(
                real_data, det_noise_precision_matrix, noise_precision_matrix
            ),
            lambda: self.NCE_zero(),
        )
        loss = loss + tf.cond(
            tf.shape(noise_data).numpy()[0] != 0,
            lambda: self.NCE_noise_data_terms(
                noise_data, nu, det_noise_precision_matrix, noise_precision_matrix
            ),
            lambda: self.NCE_zero(),
        )
        loss = loss + self.NCE_precision_matrix_term(nu, det_noise_precision_matrix)

        return loss

class Params:
    def __init__(self, shape, mask, loc=None, precision_matrix=None, only_ones=False):
        # Temporary values for testing:
        rows = shape[0]
        cols = shape[1]
        self.dim = rows * cols
        if loc is None:
            self.mu = tf.zeros((1, self.dim), dtype=tf.float64)
        else:
            self.mu = loc

        if precision_matrix is None:
            if only_ones:
                tmp = tf.Variable(np.ones((self.dim, self.dim)) * mask, dtype=tf.float64)
            else:
                arr = tf.math.abs(
                    tf.random.uniform((self.dim, self.dim), 0.1, 1, dtype=tf.float64) * mask
                )
                self.diag_precision = tf.Variable(
                    tf.linalg.band_part(arr, 0, 0), dtype=tf.float64
                )
                self.upper_precision = (
                    tf.Variable(tf.linalg.band_part(arr, 0, -1), dtype=tf.float64)
                    - self.diag_precision
                )
                self.lower_precision = tf.linalg.matrix_transpose(self.upper_precision)

                tmp = tf.Variable(
                    self.upper_precision + self.diag_precision + self.lower_precision,
                    dtype=tf.float64,
                )

            self.precision_matrix = tmp
            """try:
                np.linalg.cholesky(tmp.numpy())
                self.precision_matrix = tmp
            except:
                self.precision_matrix = []
                plt.imshow(tmp.numpy())
                plt.show()
                print("Cholesky decomposition failed.")
                raise AssertionError(
                    "The precision matrix needs to be positive definite."
                )"""
        else:
            self.precision_matrix = precision_matrix

    def get_distribution(self):
        tfp.distributions.MultivariateNormalFullCovariance(
            loc=self.mu,
            covariance_matrix=self.covariance_matrix,
            validate_args=False,
            allow_nan_stats=True,
            name="MultivariateNormalFullCovariance",
        )

    def generate_samples(self, num, slide_samples=False):
        if slide_samples:
            eps = np.random.multivariate_normal(np.zeros(self.dim),np.eye(self.dim),size=num)
            A = scipy.linalg.sqrtm(self.covariance_matrix)
            prod = np.einsum('ij,kj->ki',A,eps)

            return prod + self.mu
        else:
            return np.random.multivariate_normal(
                mean=np.squeeze(self.mu),
                cov=self.covariance_matrix,
                size=num,
                check_valid="ignore",
            ).astype(np.float32)

    @property
    def covariance_matrix(self):
        return tf.linalg.inv(self.precision_matrix)

def make_symmetric(input, sym_type="upper", mask=None):
    if sym_type == "upper":
        part = tf.linalg.band_part(input, 0, -1)
    elif sym_type == "lower":
        part = tf.linalg.band_part(input, -1, 0)
    else:
        raise ValueError

    sym = part + tf.linalg.matrix_transpose(part) - tf.linalg.band_part(part, 0, 0)

    if mask is not None:
        sym = sym * mask

    return sym




#@tf.function
def train_step(data):
    with tf.GradientTape(persistent=True) as tape:
        loss = loss_fcn(
            data,
        )

    grads = make_symmetric(tape.gradient(loss, model.precision_matrix), mask=model.mask)
    optimizer.apply_gradients(zip([grads], [model.precision_matrix]))
    return loss, grads

#@tf.function
def train_step_nce(
    real_data,
    noise_data,
    det_noise_precision_matrix,
    noise_precision_matrix,
    nu,
):
    with tf.GradientTape() as tape:
        loss = loss_fcn_NCE(
            real_data,
            noise_data,
            det_noise_precision_matrix,
            noise_precision_matrix,
            nu,
        )

    grads = make_symmetric(tape.gradient(loss, model.precision_matrix), mask=model.mask)
    optimizer.apply_gradients(zip([grads], [model.precision_matrix]))
    return loss, grads


learning_rate = 0.001
eta = 0.75  # Probability that sample is real and not noise
nu = 1 / eta - 1
batch_size = 1000
mask_type = "orthogonal"
epochs = 1
NCE = True
saving = True
decay_factor = 1


data_dictionary = get_mnist()

train_images = data_dictionary["train-images"]
train_labels = data_dictionary["train-labels"]

# train_images = np.random.uniform(0, 255, (6000, 3, 3))
train_labels = train_labels[: len(train_images)]

train_images = train_images / 255

train_images += np.random.normal(scale=1 / 100, size=np.shape(train_images))

shape = np.shape(train_images)
train_images = np.reshape(train_images, (shape[0], shape[1] * shape[2]))

pixel_wise_mean = np.mean(train_images, axis=0)

train_images = train_images - pixel_wise_mean
train_images = np.array(train_images, dtype=np.single)

train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size=batch_size)

mask = adjacency_mask(shape=(shape[1], shape[2]), mask_type=mask_type)

params = Params((shape[1], shape[2]), mask=mask, only_ones=False)
model = GaussianModel((shape[1], shape[2]), mask=mask, params=params)

noise_params = Params((shape[1], shape[2]), mask=mask, only_ones=False)
det_noise_precision_matrix = tf.linalg.det(noise_params.precision_matrix)
noise_precision_matrix = noise_params.precision_matrix


loss_fcn = model.model_loss
loss_fcn_NCE = model.NCE
optimizer = optimizers.RMSprop(learning_rate=learning_rate)

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
loss_epochs = []
loss = []
loss_diff = []
grads = []
epoch = 0

last_precision_matrix = model.precision_matrix
current_precision_matrix = model.precision_matrix
nan_precision = False
for epoch in range(epochs):
    #while True:
    print("Start of epoch ", epoch + 1)
    bar = progressbar.ProgressBar(max_value=len(train_dataset), widgets=widgets).start()
    loss_epoch = []
    for step, train_image_batch in enumerate(train_dataset):
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

            train_image_batch = tf.convert_to_tensor(tmp_real, dtype=tf.float64)
            noise_image_batch = tf.convert_to_tensor(tmp_noise, dtype=tf.float64)

            step_loss, step_grads = train_step_nce(
                train_image_batch,
                noise_image_batch,
                det_noise_precision_matrix,
                noise_precision_matrix,
                nu,
            )
        else:
            step_loss, step_grads = train_step(train_image_batch)
        if step > 1:
            last_precision_matrix = current_precision_matrix
            current_precision_matrix = model.precision_matrix

        if np.any(np.isnan(model.precision_matrix.numpy())):
            nan_precision = True
            break

        if step % 100 == 0:
            grads.append(step_grads.numpy())
            loss_epoch.append(step_loss)
        bar.update(step)

    loss_epochs.append(loss_epoch)
    loss.append(np.mean(loss_epoch))

    print("End of epoch loss: ", np.mean(loss_epoch))#, "\n")

    if saving:

        prec_name = (
            "precision_matrix_NCE_" + str(NCE) + "_epoch_" + str(epoch + 1) + ".npy"
        )
        np.save(prec_name, model.precision_matrix.numpy())

        loss_name = (
            "loss_for_each_step_NCE_"
            + str(NCE)
            + "_during_epoch_"
            + str(epoch + 1)
            + ".npy"
        )
        np.save(loss_name, loss_epoch)

    if epoch >= 1:
        loss_diff.append(loss[-1] - loss[-2])  # If loss is reducing then loss[-1]<loss[-2] --> loss[-1]-loss[-2]<0
        if loss_diff[-1] > 0:
            learning_rate = learning_rate * decay_factor
            optimizer.lr.assign(learning_rate)
            print(optimizer.get_config()["learning_rate"])
    if epoch >= 3:
        print("Mean of last three loss diffs (", loss_diff[-3:], ") is: ", np.mean(loss_diff[-3:]),"\n")
        if np.mean(loss_diff[-3:]) > 0:
            break
    if nan_precision:
        break

    epoch += 1

"""if epochs == 1:
    plt.plot(loss_epoch)
else:
    plt.plot(loss)
plt.show()"""

precision_matrix_after = model.precision_matrix.numpy()
covariance_matrix_after = model.covariance_matrix.numpy()

"""plt.imshow(precision_matrix_after, vmin=0, vmax=1)
plt.show()

plt.imshow(covariance_matrix_after)
plt.show()"""

np.save("gradients_NCE.npy", grads)
np.save("precision_matrix_before_NCE.npy", precision_matrix_before)
np.save("precision_matrix_after_NCE.npy", precision_matrix_after)

params_after_training = Params(
    (shape[1], shape[2]),
    mask,
    loc=pixel_wise_mean.flatten(),
    #loc=model.loc.numpy(),
    precision_matrix=model.precision_matrix.numpy()
)
sample = params_after_training.generate_samples(3, slide_samples=True)
sample = np.reshape(sample, (3, shape[1], shape[2]))
print(np.shape(sample))
np.save("generated_samples_NCE.npy", sample)

for smp in sample:
    plt.imshow(np.real(smp))
    plt.show()
