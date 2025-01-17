import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_image(image_path):
    """Read an image and return a one hot vector of the image"""
    img = cv2.imread(image_path)
    reshape_value = 1
    print(f"IMG SHAPE {img.shape}")

    for i in img.shape:
        reshape_value *= i

    return img.reshape((1, reshape_value)), img.shape


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
