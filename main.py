import sys

from hill import AdvHill
from image import read_image, show_image
from metrics import PSNR

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_file_name = sys.argv[1]
    else:
        raise Exception("Missing image file name")

    img, img2  = read_image(image_file_name)

    hill = AdvHill(img2)
    encrypted_img = hill.encrypt(img2.shape[0])
    print(f"ENCRYPT: {encrypted_img.shape}")
    decrypted_img = hill.decrypt(encrypted_img)
    print(f"DECRYPT: {decrypted_img.shape}")
    print(encrypted_img)

    show_image(img)
    show_image(encrypted_img)
    # psnr = PSNR(img, decryptecd_img)
    # print(psnr)
