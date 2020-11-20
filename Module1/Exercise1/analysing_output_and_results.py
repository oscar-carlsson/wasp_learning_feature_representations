import numpy as np
import matplotlib.pyplot as plt

grads = np.load('../mnist/gradients.npy')
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
plt.show()