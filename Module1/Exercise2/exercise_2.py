from unpack_data import get_mnist
import numpy as np
import tensorflow as tf
import tensorflow.keras.optimizers as optimizers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import progressbar
from generate_mask import adjacency_mask
import scipy.linalg

from exercise_2_model_and_functions import DEM
from exercise_2_model_and_functions import Params
from exercise_2_model_and_functions import whiten

"""
 IMPORTANT: Just like the presentation, I have been sloppy
 regarding the use of w or W. Both these letters refer to
 the same matrix. (Presentation maybe uses W for the matrix
 and w_k for rows in W. But let's use consistent notation.)
"""

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
