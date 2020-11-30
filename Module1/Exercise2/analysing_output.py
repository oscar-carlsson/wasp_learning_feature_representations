import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tikzplotlib as tpl

from exercise_2_model_and_functions import visualize_filters
from exercise_2_model_and_functions import DEM
from unpack_data import get_flickr30k

train_images = np.array(get_flickr30k())

train_images = train_images / 255

shape = np.shape(train_images)
train_images = np.reshape(train_images, (shape[0], shape[1] * shape[2]))

train_images = [img - np.mean(img) for img in train_images]
train_images = np.array(train_images, dtype=np.double)

empirical_cov = 1 / (shape[0] - 1) * np.einsum("ki,kj->ij", train_images, train_images)
empirical_prec = np.linalg.inv(empirical_cov)


training_saving_directory = "/home/oscar/gitWorkspaces/wasp_learning_feature_representations_module_1/Module1/Exercise2/saving_during_training/"
figure_saving_directory = "/home/oscar/gitWorkspaces/wasp_learning_feature_representations_module_1/Module1/Exercise2/output_directory/"

"""
Loading parameters
"""

gaussian_epochs = np.arange(68)+1
mask_type = "orthogonal"
DEM_epochs = np.arange(20)+1
whitening = False

"""
Saving parameters
"""
extra_axis_parameters = ["scale = 0.5"]
sigma = 1

"""
Load DEM results
"""

params = np.load(training_saving_directory+"trainable_parameters_"+str(DEM_epochs[-1])+"_epochs_whitened_"+str(whitening)+"_"+mask_type+"_mask.npy",allow_pickle=True)

param_dict = {}

for par in params:
    param_dict[par.name[0]] = par.numpy()

model = DEM.load_model(param_dict=param_dict, sigma=sigma)

#log_model = model.exponent_marginal_dist(train_images)
#log_model_mean = tf.reduce_mean(log_model)
#log_model_std = tf.math.reduce_std(log_model)



#print(log_model_mean)
#print(log_model_std)

DEM_step_loss = []
DEM_epoch_loss = []
for dem_epoch in DEM_epochs:
    step_loss = np.load(training_saving_directory+"DEM_loss_epoch_"+str(dem_epoch)+".npy")
    DEM_step_loss.append(step_loss)
    DEM_epoch_loss.append(np.mean(step_loss))

DEM_step_loss = np.concatenate(DEM_step_loss)

plt.plot(DEM_epoch_loss)
#plt.show()
tpl.save(figure_saving_directory+"dem_loss_whitening_"+str(whitening)+"_sigma_"+str(sigma)+"_mask_"+mask_type+".tex", extra_axis_parameters=extra_axis_parameters)
plt.clf()

plt.plot(DEM_step_loss)
#plt.show()
tpl.save(figure_saving_directory+"dem_loss_per_step_whitening_"+str(whitening)+"_sigma_"+str(sigma)+"_mask_"+mask_type+".tex", extra_axis_parameters=extra_axis_parameters)
plt.clf()

visualize_filters(param_dict['V'], saving=True, savepath=figure_saving_directory+"filters_in_V_whitening_"+str(whitening)+"_sigma_"+str(sigma)+"_mask_"+mask_type+".tex", extra_axis_parameters=extra_axis_parameters)

'''# This does not need to be run unless we actually rerun the Gaussian fit of the precision matrix
"""
Load Gaussian modeling of images
"""

gauss_step_loss = []
gauss_epoch_loss = []
for gauss_epoch in gaussian_epochs:
    step_loss = np.load(training_saving_directory+"loss_for_each_step_mask_"+mask_type+"_epoch_"+str(gauss_epoch)+".npy")
    gauss_step_loss.append(step_loss)
    gauss_epoch_loss.append(np.mean(step_loss))

gauss_step_loss = np.concatenate(gauss_step_loss)

plt.plot(gauss_epoch_loss)
tpl.save(figure_saving_directory+"gaussian_loss_mask_"+mask_type+".tex", extra_axis_parameters=extra_axis_parameters)
#plt.show()
plt.clf()

precision_matrix = np.load(training_saving_directory+"precision_matrix_mask_"+mask_type+"_epoch_"+str(gaussian_epochs[-1])+".npy")

plt.figure()
plt.imshow(precision_matrix)
plt.colorbar()
tpl.save(figure_saving_directory+"learned_precision_matrix_mask_"+mask_type+".tex", extra_axis_parameters=extra_axis_parameters)
#plt.show(block=False)
plt.clf()

plt.figure()
plt.imshow(empirical_prec)
#plt.title("Empirical precicion mask")
plt.colorbar()
tpl.save(figure_saving_directory+"empirical_precision_matrix.tex", extra_axis_parameters=extra_axis_parameters)
#plt.show(block=True)
plt.clf()

plt.figure()
plt.imshow(np.linalg.inv(precision_matrix))
#plt.title("Learned covariance matrix")
plt.colorbar()
tpl.save(figure_saving_directory+"learned_covariance_matrix_mask_"+mask_type+".tex", extra_axis_parameters=extra_axis_parameters)
#plt.show(block=False)
plt.clf()

plt.figure()
plt.imshow(empirical_cov)
#plt.title("Empirical covariance matrix")
plt.colorbar()
tpl.save(figure_saving_directory+"empirical_covariance_matrix.tex", extra_axis_parameters=extra_axis_parameters)
#plt.show(block=True)
plt.clf()'''