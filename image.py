import matplotlib.pyplot as plt
import numpy as np
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
    print(f"SHAPE {l, w}")
    print(img.ndim)

    #BUG: bug in (256 x 256) px RGBA
    # RGBA not working well
    n = max(l, w)
    print(f"MAX {n}")
    if n % 2:
        n += 1
    img2 = np.zeros((n, n, 3))
    print(img2.ndim)
    img2[:l, :w, :] += img
    # print(img2.shape)
    return img, img2

def show_image(img):
    plt.figure()
    plt.imshow(img)
    plt.show()