import pickle
import time
import sys
import imageio.v3 as iio


from hill import AdvHill
from image import read_image, show_image, show_images
from metrics import calculate_mse

if __name__ == "__main__":

    if len(sys.argv) > 1:
        image_file_name = sys.argv[1]
    else:
        raise Exception("Missing image file name")

    im, img  = read_image(image_file_name)

    start_encrypt = time.time()
    hill = AdvHill(img, image_file_name)
    encrypted_img = hill.encrypt(img.shape[0])
    end_encrypt = time.time()
    print(f"Waktu enkripsi {end_encrypt - start_encrypt}")
    print(f"Size real img {img.shape}")
    print(f"Size encypted img {encrypted_img.shape}")
    decrypted_img = hill.decrypt(encrypted_img)
    print(f"Size Decrypted img {decrypted_img.shape}")
    mse = calculate_mse(img, decrypted_img)
    print(f"MSE = {mse:.4f}")
    # psnr = PSNR(img, decryptecd_img)
    # print(psnr)
