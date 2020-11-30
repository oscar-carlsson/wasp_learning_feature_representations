import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tikzplotlib as tpl
import scipy.linalg

from exercise_1_model_and_functions import Params
from generate_mask import adjacency_mask

from exercise_2_model_and_functions import visualize_filters
from exercise_2_model_and_functions import DEM
from unpack_data import get_flickr30k
from unpack_data import get_flickr30k_hold_out
from unpack_data import get_mnist

training_saving_directory = "/home/oscar/gitWorkspaces/wasp_learning_feature_representations_module_1/Module1/Exercise2/saving_during_training/"
figure_saving_directory = "/home/oscar/gitWorkspaces/wasp_learning_feature_representations_module_1/Module1/Exercise2/output_directory/"

gaussian_epochs = np.arange(68) + 1
mask_type = "orthogonal"
DEM_epochs = np.arange(20) + 1
whitening = True
sigma = 1

flickr30k = np.array(get_flickr30k()) / 255
hold_out = np.array(get_flickr30k_hold_out()) / 255
mnist = get_mnist()
mnist = np.array(mnist["train-images"]) / 255

flickr30k = flickr30k - np.mean(flickr30k, axis=0)
flickr30k = np.reshape(flickr30k, (np.shape(flickr30k)[0], 28 ** 2))
hold_out = hold_out - np.mean(hold_out, axis=0)
hold_out = np.reshape(hold_out, (np.shape(hold_out)[0], 28 ** 2))
mnist = mnist - np.mean(mnist, axis=0)
mnist = np.reshape(mnist, (np.shape(mnist)[0], 28 ** 2))

dataset_precision_matrix = np.load(
    training_saving_directory
    + "precision_matrix_mask_"
    + mask_type
    + "_epoch_"
    + str(gaussian_epochs[-1])
    + ".npy"
)

if whitening:
    C = np.linalg.inv(dataset_precision_matrix)
    eigs, U = np.linalg.eig(C)

    eigs = [1 if eig < 0 else eig for eig in eigs]

    flickr30k = np.einsum(
        "ij,kj->ki",
        np.transpose(U)
        @ scipy.linalg.sqrtm(np.linalg.inv(np.diag(eigs)))
        @ np.transpose(U),
        flickr30k,
    )

    hold_out = np.einsum(
        "ij,kj->ki",
        np.transpose(U)
        @ scipy.linalg.sqrtm(np.linalg.inv(np.diag(eigs)))
        @ np.transpose(U),
        hold_out,
    )

mask = adjacency_mask(shape=(28, 28), mask_type=mask_type)

params_after_training = Params(
    (28, 28),
    mask,
    loc=np.zeros((1, 28 ** 2)),
    # loc=model.loc.numpy(),
    precision_matrix=dataset_precision_matrix,
)

dem_params = np.load(
    training_saving_directory
    + "trainable_parameters_"
    + str(DEM_epochs[-1])
    + "_epochs_whitened_"
    + str(whitening)
    + "_"
    + mask_type
    + "_mask.npy.npy",
    allow_pickle=True,
)

param_dict = {}

for par in dem_params:
    param_dict[par.name[0]] = par.numpy()

model = DEM.load_model(param_dict=param_dict, sigma=sigma)

num_samples = 10

test_flickr30k = np.reshape(
    np.array(flickr30k[:num_samples], dtype=np.double), (num_samples, 28 ** 2)
)
test_mnist = np.reshape(
    np.array(mnist[:num_samples], dtype=np.double), (num_samples, 28 ** 2)
)
test_generated_samples = np.array(
    params_after_training.generate_samples(num_samples), dtype=np.double
)
test_hold_out = np.reshape(
    np.array(hold_out[:num_samples], dtype=np.double), (num_samples, 28 ** 2)
)

log_model_train_data, log_model_hold_out, log_model_mnist, log_model_generated_data = (
    np.log(model(test_flickr30k)),
    np.log(model(test_hold_out)),
    np.log(model(test_mnist)),
    np.log(model(test_generated_samples)),
)
print("Whitening = ",whitening)

print("num_samples = ",num_samples)

print(
    "mean(log(model(train_data))): ",
    np.mean(log_model_train_data),
    "\nmean(log(model(hold_out))): ",
    np.mean(log_model_hold_out),
    "\nmean(log(model(generated_images))): ",
    np.mean(log_model_generated_data),
    "\nmean(log(model(mnist))): ",
    np.mean(log_model_mnist),
)
print(
    "std(log(model(train_data))): ",
    np.std(log_model_train_data),
    "\nstd(log(model(hold_out))): ",
    np.std(log_model_hold_out),
    "\nstd(log(model(generated_data))): ",
    np.std(log_model_generated_data),
    "\nstd(log(model(mnist))): ",
    np.std(log_model_mnist),
)
