from unpack_data import get_mnist
import numpy as np
import tensorflow as tf
import tensorflow.keras.optimizers as optimizers
import matplotlib.pyplot as plt
import progressbar
from generate_mask import adjacency_mask
import traceback

from exercise_1_model_and_functions import make_symmetric
from exercise_1_model_and_functions import Params
from exercise_1_model_and_functions import GaussianModel


@tf.function
def train_step(data):
    with tf.GradientTape(persistent=True) as tape:

        loss = loss_fcn(
            data,
        )

    grads = make_symmetric(tape.gradient(loss, model.precision_matrix), mask=model.mask)
    optimizer.apply_gradients(zip([grads], [model.precision_matrix]))
    return loss, grads


@tf.function
def train_step_nce_tf_function(
    real_data,
    noise_data,
    det_noise_precision_matrix,
    noise_precision_matrix,
    nu,
):
    with tf.GradientTape() as tape:
        loss = loss_fcn_NCE_tf_function(
            real_data,
            noise_data,
            det_noise_precision_matrix,
            noise_precision_matrix,
            nu,
        )

    grads = make_symmetric(tape.gradient(loss, model.precision_matrix), mask=model.mask)
    optimizer.apply_gradients(zip([grads], [model.precision_matrix]))
    return loss, grads

def train_step_nce(
    real_data,
    noise_data,
    det_noise_precision_matrix,
    noise_precision_matrix,
    nu,
):
    with tf.GradientTape() as tape:
        loss = loss_fcn_NCE(
            real_data,
            noise_data,
            det_noise_precision_matrix,
            noise_precision_matrix,
            nu,
        )

    grads = make_symmetric(tape.gradient(loss, model.precision_matrix), mask=model.mask)
    if not np.isnan(grads.numpy()).any():
        optimizer.apply_gradients(zip([grads], [model.precision_matrix]))
    else:
        print("Nan grads!!")
    return loss, grads

learning_rate = 0.001
eta = 0.75  # Probability that sample is real and not noise
nu = 1 / eta - 1
batch_size = 10
mask_type = "eight_neighbours"
epochs = 3
max_epoch=100
NCE = False
tf_function = False
saving = True
decay_factor = 1


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

train_images = train_images - pixel_wise_mean
train_images = tf.Variable(np.array(train_images), dtype=tf.float64)

train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size=batch_size)

mask = adjacency_mask(shape=(shape[1], shape[2]), mask_type=mask_type)

params = Params((shape[1], shape[2]), mask=mask, only_ones=False)
model = GaussianModel((shape[1], shape[2]), mask=mask, params=params)

noise_params = Params((shape[1], shape[2]), mask=mask, only_ones=False)
det_noise_precision_matrix = tf.linalg.det(noise_params.precision_matrix)
noise_precision_matrix = noise_params.precision_matrix

loss_fcn = model.model_loss
loss_fcn_NCE_tf_function = model.NCE_tf_function
loss_fcn_NCE = model.NCE
optimizer = optimizers.RMSprop(learning_rate=learning_rate)

widgets = [
    " [",
    progressbar.Timer(format="Elapsed time: %(elapsed)s"),
    "] ",
    progressbar.Bar("*"),
    " (",
    progressbar.ETA(),
    ") ",
]

precision_matrix_before = model.precision_matrix.numpy()

plt.imshow(precision_matrix_before)
plt.show()

loss_epochs = []
loss = []
loss_diff = []
grads = []
epoch = 0

last_precision_matrix = model.precision_matrix
current_precision_matrix = model.precision_matrix
nan_precision = False
nan_loss = False
# for epoch in range(epochs):
while True:
    print("Start of epoch ", epoch + 1, "\n")
    bar = progressbar.ProgressBar(max_value=len(train_dataset), widgets=widgets).start()
    loss_epoch = []
    for step, train_image_batch in enumerate(train_dataset):
        if NCE:
            train_image_batch = tf.Variable(train_image_batch)
            noise_image_batch = tf.Variable(noise_params.generate_samples(batch_size))
            noise_indicator_array = np.random.binomial(1, eta, batch_size)
            tmp_real = []
            tmp_noise = []
            for index, indicator in enumerate(noise_indicator_array):
                if indicator == 1:
                    tmp_real.append(train_image_batch[index])
                else:
                    tmp_noise.append(noise_image_batch[index])

            train_image_batch = tf.convert_to_tensor(tmp_real, dtype=tf.float64)
            noise_image_batch = tf.convert_to_tensor(tmp_noise, dtype=tf.float64)
            worked = True
            step_loss=0
            try:
                if tf_function:
                    step_loss, step_grads = train_step_nce_tf_function(
                        train_image_batch,
                        noise_image_batch,
                        det_noise_precision_matrix,
                        noise_precision_matrix,
                        nu,
                    )
                else:
                    step_loss, step_grads = train_step_nce(
                        train_image_batch,
                        noise_image_batch,
                        det_noise_precision_matrix,
                        noise_precision_matrix,
                        nu,
                    )
                #print("Step ", step, ", step loss: ", step_loss)
            except:
                traceback.print_exc()
                #print("train_image_batch shape: ", tf.shape(train_image_batch))
                #print("noise_image_batch shape: ", tf.shape(noise_image_batch))
                worked = False
                pass
        else:
            worked=True
            try:
                step_loss, step_grads = train_step(train_image_batch)
            except:
                worked=False

        if step > 1:
            last_precision_matrix = current_precision_matrix
            current_precision_matrix = model.precision_matrix

        if np.any(np.isnan(model.precision_matrix.numpy())):
            nan_precision = True
            break

        if np.isnan((step_loss)):
            nan_loss = True
            break

        if (step % 100 == 0) and worked:
            grads.append(step_grads.numpy())
            loss_epoch.append(step_loss)
        bar.update(step)

    loss_epochs.append(loss_epoch)
    loss.append(np.mean(loss_epoch))

    print("End of epoch loss: ", np.mean(loss_epoch))

    if saving:

        prec_name = (
            "precision_matrix_NCE_" + str(NCE) + "_mask_" + mask_type + "_epoch_" + str(epoch + 1) + ".npy"
        )
        np.save(prec_name, model.precision_matrix.numpy())

        loss_name = (
            "loss_for_each_step_NCE_" + str(NCE) + "_mask_" + mask_type + "_epoch_" + str(epoch + 1) + ".npy"
        )
        np.save(loss_name, loss_epoch)

    if epoch >= 1:
        loss_diff.append(
            loss[-1] - loss[-2]
        )  # If loss is reducing then loss[-1]<loss[-2] --> loss[-1]-loss[-2]<0
        if loss_diff[-1] > 0:
            learning_rate = learning_rate * decay_factor
            optimizer.lr.assign(learning_rate)
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

    if epoch >= max_epoch:
        break

    epoch += 1

"""if epochs == 1:
    plt.plot(loss_epoch)
else:
    plt.plot(loss)
plt.show()"""

precision_matrix_after = model.precision_matrix.numpy()
covariance_matrix_after = model.covariance_matrix.numpy()

"""plt.imshow(precision_matrix_after)
plt.show()

plt.imshow(precision_matrix_after-precision_matrix_before)
plt.show()

plt.imshow(covariance_matrix_after)
plt.show()"""

np.save("gradients.npy", grads)
np.save("precision_matrix_before_mask_"+mask_type+".npy", precision_matrix_before)
np.save("precision_matrix_after_mask_"+mask_type+".npy", precision_matrix_after)

params_after_training = Params(
    (shape[1], shape[2]),
    mask,
    loc=pixel_wise_mean.flatten(),
    # loc=model.loc.numpy(),
    precision_matrix=model.precision_matrix.numpy(),
)
slide_samples = np.real(params_after_training.generate_samples(3, slide_samples=True))
slide_samples = np.reshape(slide_samples, (3, shape[1], shape[2]))

np.save("generated_samples_NCE_" + str(NCE) + "_mask_"+mask_type+"_slide_method.npy", slide_samples)

"""for img in slide_samples:
    plt.imshow(img)
    plt.show()"""

samples = np.real(params_after_training.generate_samples(3, slide_samples=False))
samples = np.reshape(samples, (3, shape[1], shape[2]))

np.save("generated_samples_NCE_" + str(NCE) + "_mask_"+mask_type+"_gaussian_method.npy", slide_samples)

"""for img in samples:
    plt.imshow(img)
    plt.show()"""
