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
    print(img.shape)
    print(img2.shape)

    hill = AdvHill(img2)
    hill._generate_key(img2.shape[0])
    encrypted_img = hill.encrypt()
    print(f"ENCRYPT: {encrypted_img.shape}")
    decrypted_img = hill.decrypt(encrypted_img)
    print(f"DECRYPT: {decrypted_img.shape}")
    print(encrypted_img)

    # psnr = PSNR(img, decryptecd_img)
    # print(psnr)
