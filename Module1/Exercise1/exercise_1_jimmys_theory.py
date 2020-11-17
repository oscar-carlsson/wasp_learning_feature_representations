from unpack_data import get_mnist
import numpy as np
import tensorflow as tf
import tensorflow.keras.optimizers as optimizers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import progressbar
from generate_mask import adjacency_mask


class GaussianModel(tf.keras.layers.Layer):
    def __init__(self, shape, mask_type="orthogonal"):
        super(GaussianModel, self).__init__()
        self.mask = adjacency_mask(shape=(shape[0], shape[1]), mask_type=mask_type)
        par = Params(shape, self.mask)
        self.loc = par.mu
        self.covariance_matrix = par.cov
        self.z = tf.Variable(1, dtype=tf.float32)
        """(2 * np.pi) ** (0.5 * dim) * tf.abs(
            tf.linalg.det(self.covariance_matrix)
        ) ** 0.5"""

        dim = shape[0] * shape[1]

        arr = tf.random.uniform((dim, dim), 0.1, 1, dtype=tf.float32) * self.mask
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


class Params:
    def __init__(self, shape, mask):
        # Temporary values for testing:
        # self.cov=tf.Variable(tf.eye(dim),dtype=tf.float32) # This works
        rows = shape[0]
        cols = shape[1]
        dim = rows * cols
        self.mu = tf.zeros((1, dim), dtype=tf.float32)

        arr = np.clip(np.random.normal(size=(dim, dim)), a_min=0, a_max=None)
        # arr = np.random.uniform(0, 1, size=(dim, dim))

        self.cov = tf.Variable(
            arr * mask,
            dtype=tf.float32,
            trainable=True,
        )


eta = 0.001
epochs = 1
batch_size = 1

model = GaussianModel((28, 28), mask_type="other")

loss_fcn = model.model_loss
optimizer = optimizers.RMSprop(learning_rate=eta)


@tf.function
def train_step(data):
    with tf.GradientTape() as tape:
        loss = loss_fcn(model(data))

    grads = tape.gradient(loss, model.precision_matrix) * model.mask

    grads_diag = tf.linalg.band_part(grads, 0, 0)
    grads_upper = tf.linalg.band_part(grads, 0, -1) - grads_diag
    grads_lower = tf.linalg.matrix_transpose(grads_upper)
    grads = grads_diag + grads_upper + grads_lower

    optimizer.apply_gradients(zip([grads], [model.precision_matrix]))
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

"""# Test of forcing symmetric gradients.
with tf.GradientTape(persistent=True) as tape:
    out2 = model(train_images[:3])
    print(out2)
    loss = model.model_loss(train_images[:3])
    print(loss)

grads = tape.gradient(loss, model.precision_matrix) * model.mask
grads_diag = tf.linalg.band_part(grads, 0, 0)
grads_upper = tf.linalg.band_part(grads, 0, -1) - grads_diag
grads_lower = tf.linalg.matrix_transpose(grads_upper)
grads = grads_diag + grads_upper + grads_lower

print(grads)
mat_before = model.precision_matrix.numpy()
plt.imshow(mat_before)
plt.show()

optimizer.apply_gradients(zip([grads], [model.precision_matrix]))

mat_after = model.precision_matrix.numpy()
plt.imshow(mat_before - mat_after)
plt.show()
"""
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
        step_loss, step_grads = train_step(train_image_batch)
        # print(step_grads)
        loss.append(step_loss)
        if step % 100 == 0:
            grads.append(step_grads.numpy())
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
