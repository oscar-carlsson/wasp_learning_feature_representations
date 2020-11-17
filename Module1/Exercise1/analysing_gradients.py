import numpy as np

grads = np.load('../mnist/gradients.npy')
precision_matrix_before = np.load('../mnist/precision_matrix_before.npy')
precision_matrix_after = np.load('../mnist/precision_matrix_after.npy')

grad_diff = np.diff(grads, axis=0)
print("Grads are constant over steps: ", not np.any(grad_diff))  # =0 --> Gradient is constant over all steps