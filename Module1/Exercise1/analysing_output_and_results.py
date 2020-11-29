import numpy as np
import matplotlib.pyplot as plt
from unpack_data import get_mnist
from generate_mask import adjacency_mask

from exercise_1_model_and_functions import make_symmetric
from exercise_1_model_and_functions import Params
from exercise_1_model_and_functions import GaussianModel

import tikzplotlib as tpl

import os

'''grads = np.load('../mnist/gradients.npy')
precision_matrix_before = np.load('../mnist/precision_matrix_before.npy')
precision_matrix_after = np.load('../mnist/precision_matrix_after.npy')

grad_diff = np.diff(grads, axis=0)
print("Grads are constant over steps: ", not np.any(grad_diff))  # =0 --> Gradient is constant over all steps

previous_precision_before_crash = np.load('../mnist/previous_precision_matrix_before_crash.npy')
print(previous_precision_before_crash)
plt.imshow(previous_precision_before_crash)
plt.show()

current_precision_before_crash = np.load('../mnist/current_precision_matrix_before_crash.npy')
print(current_precision_before_crash)
plt.imshow(current_precision_before_crash)
plt.show()'''
base_path = "/home/oscar/gitWorkspaces/wasp_learning_feature_representations_module_1/Module1/Exercise1/Figures/"
NCE = False
mask_type = "eight_neighbours"
epochs = np.arange(98) + 1

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

empirical_cov = 1 / (shape[0] - 1) * np.einsum("ki,kj->ij", train_images, train_images)
empirical_prec = np.linalg.inv(empirical_cov)

# precision_after = np.load('../mnist/precision_matrix_NCE_'+str(NCE)+'_epoch_'+str(epochs[-1])+'.npy')
precision_after = np.load(
    "../mnist/precision_matrix_NCE_" + str(NCE) + "_mask_" + mask_type + "_epoch_" + str(epochs[-1]) + ".npy")

'''plt.imshow(empirical_prec-precision_after)
plt.title("Empirical precision matrix minus trained precision matrix for MNIST with mask "+mask_type)
plt.colorbar()
tpl.save(base_path+"empirical_precision_matrix_minus_trained_for_mnist_mask_"+mask_type+"_nce_"+str(NCE)+".tex")
plt.clf()

plt.imshow(empirical_prec[-28*2-14:,-28*2-14:]-precision_after[-28*2-14:,-28*2-14:])
plt.title("Last 70 elements of empirical precision matrix minus trained precision matrix for MNIST with mask "+mask_type)
plt.colorbar()
tpl.save(base_path+"last_70_elements_of_empirical_precision_matrix_minus_trained_for_mnist_mask_"+mask_type+"_nce_"+str(NCE)+".tex")
#plt.show()
plt.clf()'''

plt.imshow(empirical_prec)
plt.title("Empirical precision matrix for MNIST")
plt.colorbar()
tpl.save(base_path+"empirical_precision_matrix_minus_trained_for_mnist.tex")
plt.clf()

plt.imshow(empirical_prec[-28*2-14:,-28*2-14:])
plt.title("Last 70 elements of empirical precision matrix for MNIST")
plt.colorbar()
tpl.save(base_path+"last_70_elements_of_empirical_precision_matrix.tex")
#plt.show()
plt.clf()



loss = []
'''for epoch in epochs:
    filename = 'loss_for_each_step_NCE_'+str(NCE)+'_during_epoch_'+str(epoch)+'.npy'
    loss.append(np.load('../mnist/'+filename))'''

for epoch in epochs:
    filename = 'loss_for_each_step_NCE_' + str(NCE) + "_mask_" + mask_type + "_epoch_" + str(epoch) + ".npy"
    loss.append(np.mean(np.load('../mnist/' + filename)))

# loss = np.concatenate(loss)


plt.plot(loss)
plt.title('Loss for NCE ' + str(NCE) + ' with ' + mask_type + ' mask')
#plt.show()
tpl.save(base_path+"Loss_for_NCE_" + str(NCE) + "_with_" + mask_type + "_mask.tex")
plt.clf()


plt.imshow(precision_after)
plt.title("Precision matrix after training for NCE "+str(NCE)+" with "+mask_type+" mask")
plt.colorbar()
tpl.save(base_path+"Precision_matrix_after_training_for_NCE_"+str(NCE)+"_with_"+mask_type+"_mask.tex")
#plt.show()
plt.clf()

plt.imshow(precision_after[-28*2-14:, -28*2-14:])
plt.colorbar()
tpl.save(base_path+"Last_70_elements_Precision_matrix_after_training_for_NCE_"+str(NCE)+"_with_"+mask_type+"_mask.tex")
#plt.show()
plt.clf()



# precision_before = np.load('../mnist/precision_matrix_before_NCE.npy')

mask = adjacency_mask(shape=(shape[1], shape[2]), mask_type=mask_type)

params_after_training = Params(
    (shape[1], shape[2]),
    mask,
    loc=pixel_wise_mean.flatten(),
    # loc=model.loc.numpy(),
    precision_matrix=precision_after,
)

slide_samples = np.real(params_after_training.generate_samples(3, slide_samples=True))
slide_samples = np.reshape(slide_samples, (3, shape[1], shape[2]))

samples = np.real(params_after_training.generate_samples(3, slide_samples=False))
samples = np.reshape(samples, (3, shape[1], shape[2]))

for ind, img in enumerate(samples):
    plt.imshow(img)
    plt.title('Sample form multivariate normal\n with $\mu$=mean(pixel_val)')
    plt.colorbar()
    tpl.save(base_path+"sample_"+str(ind)+"_using_multivariate_normal_with_mu_pixel_val_for_nce_"+str(NCE)+"_with_"+mask_type+"mask.tex")
    #plt.show()
    plt.clf()

for ind, img in enumerate(slide_samples):
    plt.imshow(img)
    plt.title("Sample from $mu + \sqrt{\Lambda^{-1}}\vareps$, $\vareps \sim N(0,1)$\n with $\mu$=mean(pixel_val)")
    plt.colorbar()
    tpl.save(base_path+"sample_" + str(ind) + "_using_slide_sampling_with_mu_pixel_val_for_nce_" + str(
        NCE) + "_with_" + mask_type + "_mask.tex")
    #plt.show()
    plt.clf()

params_after_training = Params(
    (shape[1], shape[2]),
    mask,
    loc=np.zeros_like(pixel_wise_mean.flatten()),
    # loc=model.loc.numpy(),
    precision_matrix=precision_after,
)

slide_samples = np.real(params_after_training.generate_samples(3, slide_samples=True))
slide_samples = np.reshape(slide_samples, (3, shape[1], shape[2]))

samples = np.real(params_after_training.generate_samples(3, slide_samples=False))
samples = np.reshape(samples, (3, shape[1], shape[2]))

for ind, img in enumerate(samples):
    plt.imshow(img)
    plt.title('Sample form multivariate normal\n with $\mu$=0')
    plt.colorbar()
    tpl.save(base_path+"sample_" + str(ind) + "_using_multivariate_normal_with_mu_0_for_nce_" + str(
        NCE) + "_with_" + mask_type + "_mask.tex")
    #plt.show()
    plt.clf()

for ind, img in enumerate(slide_samples):
    plt.imshow(img)
    plt.title("Sample from $mu + \sqrt{\Lambda^{-1}}\vareps$, $\vareps \sim N(0,1)$\n with $\mu$=0")
    plt.colorbar()
    tpl.save(base_path+"sample_" + str(ind) + "_using_slide_sampling_with_mu_0_for_nce_" + str(
        NCE) + "_with_" + mask_type + "_mask.tex")
    #plt.show()
    plt.clf()
