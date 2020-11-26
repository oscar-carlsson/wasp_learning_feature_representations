from unpack_data import get_mnist
import numpy as np
import tensorflow as tf
import tensorflow.keras.optimizers as optimizers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import progressbar
from generate_mask import adjacency_mask
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

"""
Everything below here is untouched from Exercise 1.
Haven't started changing that code yet.
"""

data_dictionary = get_mnist()

train_images = data_dictionary["train-images"]
train_labels = data_dictionary["train-labels"]

train_images = train_images / 255

train_images += np.random.normal(scale=1 / 100, size=np.shape(train_images))

shape = np.shape(train_images)
train_images = np.reshape(train_images, (shape[0], shape[1] * shape[2]))

train_images = [img - np.mean(img) for img in train_images]
train_images = np.array(train_images, dtype=np.single)

# train_images = tf.constant(train_images, dtype=tf.float32)

model = GaussianModel((28, 28))
print(
    "Own layer output: ",
    model(tf.constant(train_images[:3], dtype=tf.float32)).numpy(),
)

eta = 0.001
epochs = 1
batch_size = 1

loss_fcn = model.model_loss
optimizer = optimizers.RMSprop(learning_rate=eta)

widgets = [
    " [",
    progressbar.Timer(format="Elapsed time: %(elapsed)s"),
    "] ",
    progressbar.Bar("*"),
    " (",
    progressbar.ETA(),
    ") ",
]

with tf.GradientTape(persistent=True) as tape:
    out2 = model(train_images[:3])
    loss = model.model_loss(train_images[:3])

grads = tape.gradient(loss, model.covariance_matrix)

print(grads)
optimizer.apply_gradients(zip([grads], [model.covariance_matrix]))

print(model.model_loss(tf.constant(train_images[:3], dtype=tf.float32)))


@tf.function
def train_step(data):
    with tf.GradientTape() as tape:
        loss = loss_fcn(model(data))

    grads = tape.gradient(loss, model.cov) * model.mask
    optimizer.apply_gradients(zip([grads], [model.cov]))
    return loss, grads


train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size=batch_size)

loss = []
for epoch in range(epochs):
    print("Start of epoch ", epoch + 1)
    bar = progressbar.ProgressBar(max_value=len(train_dataset), widgets=widgets).start()
    for step, (train_image_batch, _) in enumerate(train_dataset):
        step_loss, step_grads = train_step(train_image_batch)
        # print(step_grads)
        loss.append(step_loss)
        bar.update(step)

plt.plot(loss)
plt.show()

print(model(train_images[:3]))

arr = model.cov.numpy()
plt.imshow(arr)
plt.show()

"""distribution = model2.get_multivariate_distrubution()

sample = distribution.sample()

print(np.shape(sample))"""
