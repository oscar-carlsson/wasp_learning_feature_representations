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
gaussian_epochs_correct_mean = np.arange(67) + 1
mask_type = "orthogonal"
DEM_epochs = np.arange(20) + 1
whitening = True
sigma = 1

dataset_precision_matrix = np.load(
    training_saving_directory
    + "precision_matrix_mask_"
    + mask_type
    + "_epoch_"
    + str(gaussian_epochs[-1])
    + ".npy"
)

dataset_precision_matrix_correct_mean = np.load(
    training_saving_directory
    + "precision_matrix_correct_mean_mask_"
    + mask_type
    + "_epoch_"
    + str(gaussian_epochs_correct_mean[-1])
    + ".npy"
)

plt.figure()
plt.imshow(dataset_precision_matrix)
plt.show(block=False)

plt.figure()
plt.imshow(dataset_precision_matrix_correct_mean)
plt.show()

plt.figure()
plt.imshow(dataset_precision_matrix-dataset_precision_matrix_correct_mean)
plt.show()

print(np.max(dataset_precision_matrix-dataset_precision_matrix_correct_mean))