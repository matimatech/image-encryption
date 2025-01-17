import pickle
import sys

import cv2

from crypto import Hill
from image import read_image, show_image, show_images


def transform(np_array, shape):
    return np_array.reshape(shape).astype("uint8")


if __name__ == "__main__":

    if len(sys.argv) > 1:
        image_file_name = sys.argv[1]
    else:
        raise Exception("Missing image file name")

    img, original_shape = read_image(image_file_name)
    hill = Hill(data=img, file_name=image_file_name)

    ### Testing zone
    print(img.shape)

    # -----------------------------------------------------------------
    # ------------------------- Encoding part -------------------------
    # -----------------------------------------------------------------

    # Get the encdoed vector image
    encoded_image_vector = hill.encode(img[0])

    # Reshape to the original shape of the image
    encoded_image = encoded_image_vector.reshape(original_shape)

    # Show the decoded image
    # show_image(encoded_image.astype('uint8'))

    # Setup the encdoed file name to be used when saving the encdoed image
    img_name = image_file_name.split(".")[0]
    img_extension = image_file_name.split(".")[1]
    encoded_img_name = "{0}-encoded.{1}".format(img_name, img_extension)
    print(encoded_image.dtype)
    print(f"Image: {encoded_image}")

    # Convert to uint8
    encoded_image = encoded_image.astype("uint8")

    # Save the image
    cv2.imwrite(encoded_img_name, encoded_image)

    # Save the image as a pickle model
    pickle.dump(encoded_image_vector, open(encoded_img_name + ".pk", "wb"))

    # # -----------------------------------------------------------------
    # # ------------------------- Decoding part -------------------------
    # # -----------------------------------------------------------------

    img_vector = pickle.load(open(encoded_img_name + ".pk", "rb"))

    # Get the decoded vector image
    decoded_image_vector = hill.decode(img_vector)

    # Reshape to the original shape of the image
    decoded_image = decoded_image_vector.reshape(original_shape)

    decoded_img_name = "{0}-decoded.{1}".format(img_name, img_extension)

    # Save the image

    cv2.imwrite(decoded_img_name, decoded_image)
    show_images(encoded_image, decoded_image)
