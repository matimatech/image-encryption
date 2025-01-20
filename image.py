import numpy as np
from PIL import Image

def read_image(image_path):
    """Read an image and return a one hot vector of the image"""
    img = Image.open(image_path)
    reshape_value = 1
    print(f"IMG SHAPE {img.size}")

    # for i in img.shape:
    #     reshape_value *= i

    # return img.reshape((1, reshape_value)), img.shape

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

read_image("docs/images/lena.png")