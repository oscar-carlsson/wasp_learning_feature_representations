from unpack_data import get_mnist, get_flickr30k
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.optimizers as optimizers
import progressbar
import time

import exercise_1_model_and_functions as ex1

from generate_mask import adjacency_mask

from exercise_2_model_and_functions import DEM
from exercise_2_model_and_functions import Params
from exercise_2_model_and_functions import whiten
from exercise_2_model_and_functions import visualize_filters

import traceback
import scipy.linalg

"""
 IMPORTANT: Just like the presentation, I have been sloppy
 regarding the use of w or W. Both these letters refer to
 the same matrix. (Presentation maybe uses W for the matrix
 and w_k for rows in W. But let's use consistent notation.)
"""


"""
Setting variables for training stuff
"""
start_time = time.asctime().replace(" ", "_")
training_saving_directory = "/home/oscar/gitWorkspaces/wasp_learning_feature_representations_module_1/Module1/Exercise2/saving_during_training/"

# Variables for gaussian training
learning_rate_gaussian = 0.001
batch_size_gaussian = 100
mask_type = "orthogonal"
epochs_gaussian = 1
max_epoch_gaussian = 70
saving_gaussian_training = True
decay_factor_gaussian = 1/2

use_previous_gaussian_training = True
old_precision_matrix_name = training_saving_directory+"precision_matrix_mask_orthogonal_epoch_68.npy"

whitening = False
use_empirical_covariance = False

# Variables for DEM training
sigma=1
learning_rate_DEM = 0.001
epochs_DEM = 1
batch_size_DEM = 10
decay_factor_DEM = 1/2
max_epoch_DEM = 70
saving_DEM_training = True

"""
Load and preprocess data.
"""
train_images = np.array(get_flickr30k())

train_images = train_images / 255

shape = np.shape(train_images)
train_images = np.reshape(train_images, (shape[0], shape[1] * shape[2]))

pixel_wise_mean = np.mean(train_images, axis=0)

#train_images = [img - np.mean(img) for img in train_images]
train_images = train_images - pixel_wise_mean
train_images = np.array(train_images, dtype=np.double)

"""
Create and train gaussian model for estimating Lambda
"""
if not use_previous_gaussian_training:
    mask = adjacency_mask(shape=(28, 28), mask_type=mask_type)
    gaussian_model_params = ex1.Params((28, 28), mask=mask)

    gaussian_model = ex1.GaussianModel((28, 28), params=gaussian_model_params)

    # @tf.function
    def train_step(data):
        with tf.GradientTape(persistent=True) as tape:

            loss = loss_fcn(
                data,
            )

        grads = ex1.make_symmetric(
            tape.gradient(loss, gaussian_model.precision_matrix), mask=gaussian_model.mask
        )
        optimizer.apply_gradients(zip([grads], [gaussian_model.precision_matrix]))
        return loss, grads


    loss_fcn = gaussian_model.model_loss
    optimizer = optimizers.RMSprop(learning_rate=learning_rate_gaussian)

    widgets = [
        " [",
        progressbar.Timer(format="Elapsed time: %(elapsed)s"),
        "] ",
        progressbar.Bar("*"),
        " (",
        progressbar.ETA(),
        ") ",
    ]

    loss_epochs = []
    loss = []
    loss_diff = []
    grads = []
    epoch = 0

    nan_precision = False
    nan_loss = False
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size=batch_size_gaussian)
    #for epoch in range(epochs_gaussian):
    while True:
        print("Gaussian model: Start of epoch ", epoch + 1, "\n")
        bar = progressbar.ProgressBar(max_value=len(train_dataset), widgets=widgets).start()
        loss_epoch = []
        for step, train_image_batch in enumerate(train_dataset):

            worked = True
            try:
                step_loss, step_grads = train_step(train_image_batch)
            except:
                traceback.print_exc()
                worked = False

            if np.any(np.isnan(gaussian_model.precision_matrix.numpy())):
                nan_precision = True
                break

            if np.isnan(step_loss):
                nan_loss = True
                break

            if (step % 100 == 0) and worked:
                grads.append(step_grads.numpy())
                loss_epoch.append(step_loss)

            bar.update(step)

        loss_epochs.append(loss_epoch)
        loss.append(np.mean(loss_epoch))

        print("End of epoch loss: ", np.mean(loss_epoch))

        if saving_gaussian_training:

            prec_name = (
                training_saving_directory
		+ start_time + "_"
                + "precision_matrix_correct_mean_mask_"
                + mask_type
                + "_epoch_"
                + str(epoch + 1)
                + ".npy"
            )
            np.save(prec_name, gaussian_model.precision_matrix.numpy())

            loss_name = (
                training_saving_directory
		+ start_time + "_"
                + "loss_for_each_step_correct_mean_mask_"
                + mask_type
                + "_epoch_"
                + str(epoch + 1)
                + ".npy"
            )
            np.save(loss_name, loss_epoch)

        if epoch >= 1:
            loss_diff.append(
                loss[-1] - loss[-2]
            )  # If loss is reducing then loss[-1]<loss[-2] --> loss[-1]-loss[-2]<0
            if loss_diff[-1] > 0:
                learning_rate_gaussian = learning_rate_gaussian * decay_factor_gaussian
                optimizer.lr.assign(learning_rate_gaussian)
                print(optimizer.get_config()["learning_rate"])
        if epoch >= 3:
            print(
                "Mean of last three loss diffs (",
                loss_diff[-3:],
                ") is: ",
                np.mean(loss_diff[-3:]),
                "\n",
            )
            if np.mean(loss_diff[-3:]) > 0:
                break

        if nan_precision:
            print("NaN precision!!")
            break

        if nan_loss:
            print("NaN loss!!")
            break

        if epoch >= max_epoch_gaussian:
            break

        epoch += 1

    dataset_precision_matrix = gaussian_model.precision_matrix.numpy()
else:
    dataset_precision_matrix = np.load(old_precision_matrix_name)
"""
Data whitening
"""

if whitening:
    if use_empirical_covariance:
        C = 1 / (shape[0] - 1) * np.einsum("ki,kj->ij", train_images, train_images)
    else:
        C = np.linalg.inv(dataset_precision_matrix)
    eigs, U = np.linalg.eig(C)

    eigs = [1 if eig < 0 else eig for eig in eigs]

    train_images = np.einsum(
        "ij,kj->ki",
        np.transpose(U)
        @ scipy.linalg.sqrtm(np.linalg.inv(np.diag(eigs)))
        @ np.transpose(U),
        train_images,
    )

"""
Create and train DEM on whitened data
"""

model = DEM((28, 28), sigma=sigma)  # Testing defining a model

loss_fcn = model.score_matching
optimizer = optimizers.RMSprop(learning_rate=learning_rate_DEM)

widgets = [
    " [",
    progressbar.Timer(format="Elapsed time: %(elapsed)s"),
    "] ",
    progressbar.Bar("*"),
    " (",
    progressbar.ETA(),
    ") ",
]


@tf.function
def train_step(data):
    with tf.GradientTape() as tape:
        loss = loss_fcn(data)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, grads


train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size=batch_size_DEM)


loss = []
loss_diff = []
epoch = 0
#for epoch in range(epochs_DEM):
while True:
    print("DEM: Start of epoch ", epoch + 1)
    bar = progressbar.ProgressBar(max_value=len(train_dataset), widgets=widgets).start()
    epoch_loss = []
    for step, train_image_batch in enumerate(train_dataset):
        step_loss, step_grads = train_step(train_image_batch)
        # print(step_grads)
        epoch_loss.append(step_loss)
        bar.update(step)

    loss.append(np.mean(epoch_loss))

    print("End of epoch loss: ", np.mean(epoch_loss))

    if saving_DEM_training:
        model.save(training_saving_directory,other="_"+str(epoch+1)+"_epochs_whitened_"+str(whitening)+"_"+mask_type+"_mask")
        np.save(training_saving_directory
		+ start_time
		+ "_DEM_loss_epoch_"
		+ str(epoch+1)
		+ "_whitened_"
		+ str(whitening) + "_"
		+ mask_type
		+ "_mask.npy",
		epoch_loss)
    if epoch >= 1:
        loss_diff.append(
            loss[-1] - loss[-2]
        )  # If loss is reducing then loss[-1]<loss[-2] --> loss[-1]-loss[-2]<0
        if loss_diff[-1] > 0:
            learning_rate_DEM = learning_rate_DEM * decay_factor_DEM
            optimizer.lr.assign(learning_rate_DEM)
            print(optimizer.get_config()["learning_rate"])
    if epoch >= 3:
        print(
            "Mean of last three loss diffs (",
            loss_diff[-3:],
            ") is: ",
            np.mean(loss_diff[-3:]),
            "\n",
        )
        if np.mean(loss_diff[-3:]) > 0:
            break

    if epoch >= max_epoch_DEM:
        break

    epoch += 1

plt.plot(loss)
plt.show()

print(model(train_images[:3]))

visualize_filters(model.V.numpy(), background_val=-10)

'''
"""
Model testing
"""
whitening = True


with tf.GradientTape() as tape:
    loss = tf.reduce_sum(
        model.exponent_marginal_dist(train_images[:3])
    )  # A mockup loss function

grad_log = model.grad_log_DEM(train_images[:3])
print(grad_log)

laplace_log = model.laplace_log_DEM(train_images[:3])
print(laplace_log)
with tf.GradientTape() as tape:
    loss = model.score_matching(train_images[:3])
print(loss)

grads = tape.gradient(
    loss, model.trainable_variables
)  # Get gradients of loss wrt [self.b, self.c, self.V, self.W]
print(grads)'''
