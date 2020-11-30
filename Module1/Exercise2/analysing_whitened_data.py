import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tikzplotlib as tpl
import scipy.linalg

from exercise_2_model_and_functions import visualize_filters
from exercise_2_model_and_functions import DEM
from unpack_data import get_flickr30k

training_saving_directory = "/home/oscar/gitWorkspaces/wasp_learning_feature_representations_module_1/Module1/Exercise2/saving_during_training/"
figure_saving_directory = "/home/oscar/gitWorkspaces/wasp_learning_feature_representations_module_1/Module1/Exercise2/output_directory/"
gaussian_epochs = np.arange(68)+1
mask_type = "orthogonal"


train_images = np.array(get_flickr30k())

train_images = train_images / 255

shape = np.shape(train_images)
train_images = np.reshape(train_images, (shape[0], shape[1] * shape[2]))

train_images = [img - np.mean(img) for img in train_images]
train_images = np.array(train_images, dtype=np.double)

empirical_cov = 1 / (shape[0] - 1) * np.einsum("ki,kj->ij", train_images, train_images)
empirical_prec = np.linalg.inv(empirical_cov)

dataset_precision_matrix = np.load(training_saving_directory+"precision_matrix_mask_"+mask_type+"_epoch_"+str(gaussian_epochs[-1])+".npy")

C = np.linalg.inv(dataset_precision_matrix)
eigs, U = np.linalg.eig(C)

eigs = [1 if eig < 0 else eig for eig in eigs]

train_images_whitened_trained_precision = np.einsum(
    "ij,kj->ki",
    U#np.transpose(U)
    @ scipy.linalg.sqrtm(np.linalg.inv(np.diag(eigs)))
    @ U,#np.transpose(U),
    train_images,
)

for ind, img in enumerate(train_images_whitened_trained_precision[:3]):
    plt.imshow(np.reshape(img,(28,28)))
    plt.show()
    #tpl.save(figure_saving_directory+"img_"+str(ind)+"_whitened_used_trained_covariance.tex",extra_axis_parameters = ["scale = 0.5"])
    plt.clf()




C = empirical_cov
eigs, U = np.linalg.eig(C)

eigs = [1 if eig < 0 else eig for eig in eigs]

train_images_whitened_empirical_precision = np.einsum(
    "ij,kj->ki",
    U#np.transpose(U)
    @ scipy.linalg.sqrtm(np.linalg.inv(np.diag(eigs)))
    @ U,#np.transpose(U),
    train_images,
)


for ind, img in enumerate(train_images_whitened_empirical_precision[:3]):
    plt.imshow(np.reshape(img,(28,28)))
    plt.show()
    #tpl.save(figure_saving_directory+"img_"+str(ind)+"_whitened_used_empirical_covariance.tex",extra_axis_parameters = ["scale = 0.5"])
    plt.clf()

for ind, img in enumerate(train_images[:3]):
    plt.imshow(np.reshape(img,(28,28)))
    #tpl.save(figure_saving_directory + "img_" + str(ind) + "_non_whitened.tex",
    #         extra_axis_parameters=["scale = 0.5"])
    plt.show()
    plt.clf()