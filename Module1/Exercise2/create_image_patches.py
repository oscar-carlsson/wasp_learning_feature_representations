import os
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
from PIL import Image

dir = os.path.abspath("Images/")

k = 0
for filename in os.listdir(dir):
    # Open image and convert to 8 bit grayscale
    image = Image.open(os.path.join(dir, filename)).convert('L')
    # image.show()

    # Create appropriately sized array from image
    rows = image.size[1]
    cols = image.size[0]
    image_array = np.zeros((1, rows, cols, 1))
    image_array[0, :, :, 0] = np.array(image)

    # Normalize to [0,1] range
    #image_array = image_array / 255

    # Extract many image patches which are strides apart in
    # each dim. Output has shape (1, X, Y, 768) where X, Y are
    # number of patches in each direction (up/down, left/right)
    image = tf.image.extract_patches(image_array,
                                     sizes=[1, 28, 28, 1],
                                     strides=[1, 56, 56, 1],
                                     rates=[1, 6, 6, 1],
                                     padding="VALID")


    # Select 2 random patches per image
    rows = image.shape[1]
    cols = image.shape[2]
    if rows == 0 or cols == 0:
        continue # Avoid problematic cases by skipping the current image

    m = np.random.choice(rows, size=2)
    n = np.random.choice(cols, size=2)
    im1 = image[0, m[0], n[0], :].numpy().reshape((28, 28))
    im2 = image[0, m[1], n[1], :].numpy().reshape((28, 28))

    # Save patches as images
    name = "Patches/patch%d.jpeg"%(k,)
    im1 = Image.fromarray(im1)
    im1 = im1.convert('RGB')
    im1.save(name)

    name = "Patches/patch%d.jpeg"%(k+1,)
    im2 = Image.fromarray(im2)
    im2 = im2.convert('RGB')
    im2.save(name)

    """
    plt.imshow(im1, cmap='gray', vmin=0, vmax=255)
    plt.show()

    plt.imshow(im2, cmap='gray', vmin=0, vmax=255)
    plt.show()
    """

    k += 2
    if k == 10: # Temp value for testing. Change to 50000 later.
        break
