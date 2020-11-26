import numpy as np
import matplotlib.pyplot as plt
from unpack_data import get_mnist
from generate_mask import adjacency_mask

from exercise_1_model_and_functions import make_symmetric
from exercise_1_model_and_functions import Params
from exercise_1_model_and_functions import GaussianModel

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

NCE = False

epochs = np.arange(100)+1
loss = []
for epoch in epochs:
    filename = 'loss_for_each_step_NCE_'+str(NCE)+'_during_epoch_'+str(epoch)+'.npy'
    loss.append(np.load('../mnist/'+filename))

loss = np.concatenate(loss)

plt.plot(loss)
plt.show()


precision_after = np.load('../mnist/precision_matrix_NCE_'+str(NCE)+'_epoch_'+str(epochs[-1])+'.npy')
plt.imshow(precision_after)
plt.show()

precision_before = np.load('../mnist/precision_matrix_before_NCE.npy')

mask = adjacency_mask(shape=(shape[1], shape[2]), mask_type="orthogonal")

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

for img in samples:
    plt.imshow(img)
    plt.title('Sample form multivariate normal\n with mu=mean(pixel_val)')
    plt.show()

for img in slide_samples:
    plt.imshow(img)
    plt.title('Sample from mu + \sqrt{\lambda^{-1}}*eps, eps sampled from N(0,1)\n with mu=mean(pixel_val)')
    plt.show()

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

for img in samples:
    plt.imshow(img)
    plt.title('Sample form multivariate normal\n with mu=0')
    plt.show()

for img in slide_samples:
    plt.imshow(img)
    plt.title('Sample from mu + \sqrt{\lambda^{-1}}*eps, eps sampled from N(0,1)\n with mu=0')
    plt.show()