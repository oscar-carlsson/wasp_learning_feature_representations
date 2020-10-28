from unpack_data import get_mnist
import numpy as np
import matplotlib.pyplot as plt

data_dictionary = get_mnist()

train_images = data_dictionary["train-images"]

train_images = train_images / 255

train_images += np.random.normal(scale=1 / 100, size=np.shape(train_images))

shape = np.shape(train_images)
print(shape)

train_images = np.reshape(train_images, (shape[0], shape[1] * shape[2]))

train_images = [img - np.mean(img) for img in train_images]


class p:
    def __init__(self, vector_length):
        self.vector_length = vector_length
        self.Z = 1
        self.covariance_matrix = np.random.random((vector_length, vector_length))

    def __call__(self, data):
        return (
            1
            / self.Z
            * np.exp(-1 / 2 * np.matmul(data, np.matmul(self.covariance_matrix, data)))  # This works for just one row vector, redo for multiple
        )

p_theta = p(28*28)

print(p_theta(train_images))