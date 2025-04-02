import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio.v3 as iio

def read_image(image_path):
    """
    Read an image and return a one hot vector of the image
    If size of an image is odd, transform the img to even size image

    Parameter:
    image_path: path of the image

    Return:
    img2: Transformed image object
    """
    img = iio.imread(image_path)
    l = img.shape[0]
    w = img.shape[1]

    n = max(l, w)
    if n % 2:
        n += 1
    img2 = np.zeros((n, n, 3))
    img2[:l, :w, :] += img
    # print(img)
    return img, img2


def show_image(image):
    """Show a single image"""
    plt.imshow(image)
    plt.show()


def show_images(a, b):
    """Show two images side by side"""
    plot_image = np.concatenate((a, b), axis=1)
    plt.imshow(plot_image)
    while 1:
        plt.show()

# read_image("docs/images/lena.png")