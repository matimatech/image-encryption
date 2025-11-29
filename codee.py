# Python implementation: Advance Hill Cipher (3x3) + ElGamal key encryption
# - Encrypts/decrypts a 256x256 jpg RGB image using a 3x3 Hill cipher key modulo 256.
# - Encrypts the 3x3 key matrix entries using a simple ElGamal scheme over prime p=65537.
# - Saves encrypted image and decrypted image to /mnt/data
# Usage: set input_path to your jpg image (will be resized to 256x256 and converted to RGB if needed).
# Note: This is a teaching/demo implementation. For real cryptographic use, use vetted libraries & parameters.
from PIL import Image
import numpy as np
import os
import math
import random
import time
import sys
from skimage.metrics import structural_similarity as ssim

def set_global_seed(seed=42):
    import os
    import random
    import numpy as np

    # Set seed untuk semua sumber randomness umum
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)



set_global_seed(42)

# ---------- Utilities for modular arithmetic ----------
def egcd(a, b):
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)

def modinv(a, m):
    g, x, _ = egcd(a % m, m)
    if g != 1:
        raise ValueError(f"No modular inverse for {a} mod {m} (gcd != 1)")
    return x % m

def extended_gcd(a, b):
    """
    Implementasi Extended Euclidean Algorithm.
    Mengembalikan (gcd, x, y) sedemikian rupa sehingga: a*x + b*y = gcd
    """
    if a == 0:
        return (b, 0, 1)
    else:
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return (gcd, x, y)

def mod_inverse(a, m):
    """
    Mencari modular multiplicative inverse dari a modulo m.
    (yaitu a^-1 mod m)
    """
    # Memastikan a positif
    a = a % m
    gcd, x, y = extended_gcd(a, m)
    if gcd != 1:
        # Invers tidak ada
        raise ValueError(f"Modular inverse does not exist for {a} mod {m} (gcd is {gcd})")
    else:
        return x % m

def generate_involutory_key_manual(A22_scalar, k, mod=256):
    """
    Membuat kunci involutori 2x2 berdasarkan input manual (skalar)
    sesuai algoritma Acharya/Azam.
    """
    print(f"--- Membuat Kunci Manual 2x2 (A22={A22_scalar}, k={k}, mod={mod}) ---")
    
    # Langkah 2: Hitung A11
    A11 = (-A22_scalar) % mod
    print(f"A11 = -{A22_scalar} % {mod} = {A11}")
    
    # Langkah 4 & 5: Hitung k_inv
    if math.gcd(k, mod) != 1:
        raise ValueError(f"k={k} not coprime to mod={mod}, inverse does not exist")
    k_inv = mod_inverse(k, mod)
    print(f"k = {k}, k_inv = {k_inv}")
    
    # Langkah 6: Hitung A12
    # A12 = k(I - A11) mod N
    A12 = (k * (1 - A11)) % mod
    print(f"A12 = {k} * (1 - {A11}) % {mod} = {A12}")
    
    # Langkah 7: Hitung A21
    # A21 = k^-1(I + A11) mod N
    A21 = (k_inv * (1 + A11)) % mod
    print(f"A21 = {k_inv} * (1 + {A11}) % {mod} = {A21}")
    
    # Langkah 8: Bentuk Matriks K
    K = np.array([[A11, A12], [A21, A22_scalar]], dtype=np.int64)
    return K

def generate_involutory_key_random(n, mod=256):
    """
    Membuat kunci involutori nxn acak (n harus genap)
    sesuai algoritma Acharya/Azam.
    """
    if n % 2 != 0:
        raise ValueError("n must be even for this algorithm.")
        
    print(f"\n--- Membuat Kunci Acak {n}x{n} (mod={mod}) ---")
    sub_size = n // 2
    
    # Langkah 1: Buat A22 acak
    A22 = np.random.randint(0, mod, size=(sub_size, sub_size), dtype=np.int64)
    
    # Langkah 2: Hitung A11
    A11 = (-A22) % mod
    
    # Matriks Identitas
    I = np.identity(sub_size, dtype=np.int64)
    
    # Langkah 3 & 4: Cari skalar k acak yang punya invers
    while True:
        k = random.randint(1, mod - 1)
        if math.gcd(k, mod) == 1:
            break
    # k=7
    print(f"k = {k} \n")
    # Langkah 5: Hitung k_inv
    k_inv = mod_inverse(k, mod)
    print(f"k inverse = {k_inv} \n")
    
    # Langkah 6: Hitung A12
    A12 = (k * (I - A11)) % mod
    
    # Langkah 7: Hitung A21
    A21 = (k_inv * (I + A11)) % mod
    
    # Langkah 8: Bentuk Matriks K
    print(f"A11 = {A11} \n A12 = {A12} \n A21 = {A21} \n A22 = {A22} \n")
    K = np.block([[A11, A12], [A21, A22]])
    return K

def validate_key(K, mod=256):
    """Memeriksa apakah K^2 = I mod (mod)"""
    print(f"\nMemvalidasi matriks K:\n{K}")
    K_squared = (K @ K) % mod
    I = np.identity(K.shape[0], dtype=np.int64)
    
    print(f"K^2 % {mod}:\n{K_squared}")
    if np.array_equal(K_squared, I):
        print("Validasi Berhasil: K^2 = I (K adalah involutori)")
    else:
        print("Validasi Gagal: K^2 != I")
    
    # Cek Invertible
    det_K = int(round(np.linalg.det(K))) % mod
    print(f"det(K) % {mod} = {det_K}")
    if math.gcd(det_K, mod) == 1:
        print("Validasi Berhasil: det(K) coprime dengan {mod} (K adalah invertible)")
    else:
        print("Validasi Gagal: det(K) tidak coprime dengan {mod}")
    return K_squared

# ---------- ElGamal simple implementation over prime p ----------
# We'll use p = 65537 (a Fermat prime), generator g=3 (commonly used small generator)
# Note: In practice, choose large safe primes and validated generators.
p = 65537
g = 3

def elgamal_keygen():
    x = random.randrange(1, p-2)  # private key
    y = pow(g, x, p)             # public key y = g^x mod p
    print(f"GGG = {g}")
    print(f"xxxx = {x}")
    print(f"YYY = {y}")
    return (p, g, y), x

def elgamal_encrypt(m, pubkey):
    """
    p	: Bilangan prima.
    g 	: generator (akar primitif dari bilangan prima).
    y   : kunci publik
    m	: Plaintext.
	k   : bilangan acak di sekitar 1≤k≤p-2. 
    """
    p, g, y = pubkey
    if not (0 <= m < p):
        # map m into range by reducing mod p
        m = m % p
    k = random.randrange(1, p-2)
    a = pow(g, k, p)
    b = (m * pow(y, k, p)) % p
    print(f"y = {y}, m = {m}, k = {k}, a = {a}, b = {b}, p = {p} \n")
    return (a, b)

def elgamal_decrypt(cipher, privkey):
    c1, c2 = cipher
    x = privkey
    s = pow(c1, x, p)
    s_inv = modinv(s, p)
    m = (c2 * s_inv) % p
    print(f"mmm = {m}")
    return m

# ---------- Hill cipher operations on RGB image ----------
MOD = 256  # byte-wise arithmetic
KEY_SIZE = 3

def encrypt_block(P, K, mod=256):
    """
    Menerapkan rumus enkripsi AdvHill pada satu blok P.
    Rumus: C = K * (K * P)^T mod 256
    """
    
    # 1. Enkripsi Level Pertama: P_temp = K * P
    # Kita perlu memastikan tipe data adalah int64 agar tidak overflow
    # saat perkalian sebelum modulo.
    P_temp = np.dot(K, P.astype(np.int64)) % mod
    # print(f"KKKKK = {K}")
    # print(f"PPPPP = {P}")
    # print(f"TEMPPP = {P_temp}")
    # 2. Operasi Transpose: P_trans = (P_temp)^T
    P_trans = P_temp.T
    # print(f"TRANSPOSE= {P_trans}")
    
    # 3. Enkripsi Level Kedua: C = K * P_trans
    C_block = np.dot(K, P_trans) % mod
    # print(f"CCCC = {C_block}")
    
    return C_block

def hill_encrypt_image(image_array, image_path, K, block_size):
    # start = time.time()
    # h, w, ch = img_array.shape
    # assert ch == 3 and K.shape == (n,n)
    # out = np.zeros_like(img_array, dtype=np.uint8)
    # # for each pixel, treat RGB as vector length 3 and multiply K @ v mod 256
    # flat = img_array.reshape(-1, n).astype(int)
    # enc = (flat.dot(K.T) % MOD).astype(np.uint8)
    # out = enc.reshape(h, w, 3)
    # end = time.time()
    # print(f"Waktu enkripsi citra digital = {end - start}")
    # return out
    try:
        # start = time.time()

        height, width, channels = image_array.shape
        print(f"Citra asli dimuat: {image_path} (Ukuran: {width}x{height})")
        
        # Buat array kosong untuk hasil enkripsi
        encrypted_array = np.zeros_like(image_array, dtype=np.uint8)

        # Proses setiap kanal (R, G, B) secara terpisah 
        for c_idx in range(channels):
            channel_data = image_array[:, :, c_idx]
            
            # --- Penanganan Padding ---
            # Hitung padding yang diperlukan agar dimensi habis dibagi block_size
            h_pad = (block_size - height % block_size) % block_size
            w_pad = (block_size - width % block_size) % block_size
            
            # Terapkan padding (dengan nilai 0)
            padded_channel = np.pad(channel_data, ((0, h_pad), (0, w_pad)), 'constant', constant_values=0)
            
            new_height, new_width = padded_channel.shape
            encrypted_channel = np.zeros_like(padded_channel, dtype=np.int64)
            
            # --- Iterasi per Blok ---
            print(f"Memproses Kanal {c_idx} (R,G,B)...")
            for r in range(0, new_height, block_size):
                for c in range(0, new_width, block_size):
                    # Ambil blok plaintext 4x4
                    P_block = padded_channel[r : r + block_size, c : c + block_size]
                    
                    # Enkripsi blok tersebut
                    C_block = encrypt_block(P_block, K, MOD)
                    
                    # Simpan blok ciphertext
                    encrypted_channel[r : r + block_size, c : c + block_size] = C_block
                
            # --- Hapus Padding ---
            # Potong kembali array ke ukuran aslinya
            encrypted_array[:, :, c_idx] = encrypted_channel[0:height, 0:width].astype(np.uint8)
        arr = encrypted_array    
        # Pisahkan kanal warna: Red, Green, Blue
        R = arr[0:4, 0:4, 0]  # Kanal merah
        G = arr[0:4, 0:4, 1]  # Kanal hijau
        B = arr[0:4, 0:4, 2]  # Kanal biru
        # print(f"REDD {G}")
        np.savetxt("kanal_enc_R.txt", R, fmt="%d", delimiter=",")
        np.savetxt("kanal_enc_G.txt", G, fmt="%d", delimiter=",")
        np.savetxt("kanal_enc_B.txt", B, fmt="%d", delimiter=",")

        
        print(f"\nProses enkripsi selesai.")
        # end = time.time()
        # print(f"Waktu enkripsi citra digital = {end - start}")

        return encrypted_array

    except FileNotFoundError:
        print(f"ERROR: File '{image_path}' tidak ditemukan.")
        return None
    except Exception as e:
        print(f"Terjadi error: {e}")
        return None

def process_advhill_block(block, K, mod=256):
    """
    Menerapkan rumus AdvHill: Output = K * (K * Input)^T mod 256
    Fungsi ini identik untuk enkripsi dan dekripsi
    karena kunci K adalah involutori.
    """
    
    # 1. Transformasi Level Pertama: Temp = K * Input
    # Kita perlu memastikan tipe data adalah int64 agar tidak overflow
    # saat perkalian sebelum modulo.
    Temp_block = np.dot(K, block.astype(np.int64)) % mod
    # print(f"Decrypt temp = {Temp_block}")
    # 2. Operasi Transpose: Trans_block = (Temp)^T
    Trans_block = Temp_block.T
    # print(f"Decrypt Tranpose = {Trans_block}")
    
    # 3. Transformasi Level Kedua: Output = K * Trans_block
    Output_block = np.dot(K, Trans_block) % mod
    # print(f"Decrypt output = {Output_block}")
    
    return Output_block

def decrypt_image_advhill(img, encrypted_image_path, K, block_size=4):    # start = time.time()
    # h, w, _ = enc_array.shape
    # flat = enc_array.reshape(-1, n).astype(int)
    # dec = (flat.dot(K_inv.T) % MOD).astype(np.uint8)
    # out = dec.reshape(h, w, 3)
    # end = time.time()
    # print(f"Waktu dekripsi citra digital = {end - start}")
    # return out

    """
    Mendekripsi seluruh citra berwarna menggunakan algoritma AdvHill.
    original_size adalah tuple (height, width) dari citra asli.
    """
    mod = 256
    
    try:
        # Membuka citra terenkripsi dan mengonversi ke array NumPy
        encrypted_array = np.array(img)
        
        # Mengambil ukuran asli (penting untuk memotong padding)
        height, width = img.size
        print(f"Citra terenkripsi dimuat: {encrypted_image_path}")
        
        # Buat array kosong untuk hasil dekripsi
        decrypted_array = np.zeros((height, width, 3), dtype=np.uint8)

        # Proses setiap kanal (R, G, B) secara terpisah
        for c_idx in range(3): # 3 kanal
            channel_data = encrypted_array[:, :, c_idx]
            
            # --- Penanganan Padding (Sama seperti enkripsi) ---
            # Kita harus menambahkan padding yang sama seperti saat enkripsi
            # agar blok 4x4-nya pas.
            h_pad = (block_size - height % block_size) % block_size
            w_pad = (block_size - width % block_size) % block_size
            
            padded_channel = np.pad(channel_data, ((0, h_pad), (0, w_pad)), 'constant', constant_values=0)
            
            new_height, new_width = padded_channel.shape
            decrypted_channel = np.zeros_like(padded_channel, dtype=np.int64)
            
            # --- Iterasi per Blok ---
            print(f"Memproses Kanal {c_idx} (R,G,B)...")
            for r in range(0, new_height, block_size):
                for c in range(0, new_width, block_size):
                    # Ambil blok ciphertext 4x4
                    C_block = padded_channel[r : r + block_size, c : c + block_size]
                    
                    # Dekripsi blok tersebut (rumus identik)
                    P_block = process_advhill_block(C_block, K, mod)
                    
                    # Simpan blok plaintext
                    decrypted_channel[r : r + block_size, c : c + block_size] = P_block
 
            # --- Hapus Padding ---
            # Potong kembali array ke ukuran aslinya
            decrypted_array[:, :, c_idx] = decrypted_channel[0:height, 0:width].astype(np.uint8)
            
        # Konversi array NumPy kembali ke objek Gambar PIL
        decrypted_image = Image.fromarray(decrypted_array, 'RGB')
        
        print(f"\nProses dekripsi selesai.")
        
        return decrypted_array

    except FileNotFoundError:
        print(f"ERROR: File '{encrypted_image_path}' tidak ditemukan.")
        return None
    except Exception as e:
        print(f"Terjadi error: {e}")
        return None
# ---------- Helper: load & prepare image ----------
def load_prepare_image(path):
    img = Image.open(path).convert('RGBA')  # read with possible alpha
    # discard alpha if present by converting to RGB explicitly
    img = img.convert('RGB')
    # if img.size != size:
    #     img = img.resize(size, Image.BICUBIC)
    arr = np.array(img, dtype=np.uint8)
    # # Pisahkan kanal warna: Red, Green, Blue
    # R = arr[0:4, 0:4, 0]  # Kanal merah
    # G = arr[0:4, 0:4, 1]  # Kanal hijau
    # B = arr[0:4, 0:4, 2]  # Kanal biru
    # print(f"REDD {G}")
    # np.savetxt("kanal_asli_R.txt", R, fmt="%d", delimiter=",")
    # np.savetxt("kanal_asli_G.txt", G, fmt="%d", delimiter=",")
    # np.savetxt("kanal_asli_B.txt", B, fmt="%d", delimiter=",")
    return arr

def save_image_from_array(arr, path):
    im = Image.fromarray(arr, mode='RGB')
    im.save(path)

# ---------- Evaluation Metrics ----------
def mse(img1, img2):
    """Compute Mean Square Error between two RGB images."""
    diff = np.subtract(img1.astype(np.float64), img2.astype(np.float64))
    err = np.mean(np.square(diff))
    return err

def psnr(img1, img2):
    """Compute PSNR between two RGB images."""
    mse_val = mse(img1, img2)
    if mse_val == 0:
        return float('inf')
    max_pixel = 255.0
    psnr_val = 10 * math.log10((max_pixel ** 2) / mse_val)
    return psnr_val

def ssim_index(img1, img2):
    """Compute SSIM (mean over RGB channels)."""
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")
    ssim_total = 0
    for i in range(3):  # for each channel
        ssim_total += ssim(img1[:,:,i], img2[:,:,i], data_range=255)
    return ssim_total / 3

# ---------- Demo pipeline ----------
def demo_pipeline(input_path='input.jpg', ukuran_kunci=4):
    # prepare image
    if not os.path.exists(f"input/{input_path}"):
        # create simple test image if not exists
        print("Input not found; creating a sample image at /mnt/data/input.jpg")
        img = Image.new('RGB', (256,256))
        for x in range(256):
            for y in range(256):
                img.putpixel((x,y), (x, y, (x+y)//2))
        img.save('output/input.jpg')
        input_path = 'output/input.jpg'
    start = time.time()

    n = ukuran_kunci
    img = load_prepare_image(f"input/{input_path}")

    print("Loaded image shape:", img.shape)
    end = time.time()
    print(f"RES {end - start}")
    # generate Hill key
    K = generate_involutory_key_random(n, mod=256)
    validate_key(K)
    # ElGamal keygen and encrypt each key entry
    pubkey, privkey = elgamal_keygen()
    print("ElGamal public key (p,g,y):", pubkey)
    print(f"ElGamal Privat Key = {privkey}")
    # encrypt key entries (store as list of tuples)
    start = time.time()
    encrypted_key = [[elgamal_encrypt(int(K[i,j]), pubkey) for j in range(n)] for i in range(n)]
    print(f"Encrypted Key = {encrypted_key}")

    print("Encrypted key entries (c1,c2) stored. Example entry [0,0]:", encrypted_key[0][0])

    # encrypt image using Hill
    enc_array = hill_encrypt_image(img, input_path, K, n)
    end = time.time()
    enc_path = f'output/encrypted/{input_path.split('.')[0]}_encrypted_{ukuran_kunci}.jpg'

    encrypted_image = Image.fromarray(enc_array, 'RGB')

    encrypted_image.save(enc_path)

    print("Encrypted image saved to", enc_path)
    print("Waktu enkripsi:", end - start, "detik")

    # decrypt key entries with ElGamal (simulate receiver)
    start = time.time()
    decrypted_key = np.zeros((n,n), dtype=int)
    for i in range(n):
        for j in range(n):
            decrypted_key[i,j] = elgamal_decrypt(encrypted_key[i][j], privkey)
    print("Decrypted key entries (mod p) recovered:\n", decrypted_key)
    # since entries were <256, they match exactly
    # compute K_inv again to decrypt image (mod 256)
    K_recovered = decrypted_key % MOD
    # decrypt image
    dec_array = decrypt_image_advhill(encrypted_image, enc_path, K_recovered, block_size=n)
    end = time.time()
    dec_path = f'output/decrypted/{input_path.split('.')[0]}_decrypted_{ukuran_kunci}.jpg'
    # save_image_from_array(dec_img, dec_path)
    decrypted_image = Image.fromarray(dec_array, 'RGB')

    decrypted_image.save(dec_path)
    print("Waktu dekripsi:", end - start, "detik")
    print("Decrypted image saved to", dec_path)

    orig = np.array(Image.open(f"input/{input_path}").convert('RGB'))
    decrypted = np.array(Image.open(dec_path).convert('RGB'))
    encrypted = np.array(Image.open(enc_path).convert('RGB'))

        # ---------- Evaluate ----------
    print(np.array_equal(orig, decrypted))
    mse_val = mse(orig, decrypted)
    psnr_val = psnr(orig, decrypted)
    ssim_val = ssim_index(orig, decrypted)

    print("=== Evaluation Results (Original vs Decrypted) ===")
    print(f"MSE  : {mse_val:.6f}")
    print(f"PSNR : {psnr_val:.2f} dB")
    print(f"SSIM : {ssim_val:.4f}")

    # Optional: compare Original vs Encrypted (should be low similarity)
    mse_enc = mse(orig, encrypted)
    psnr_enc = psnr(orig, encrypted)
    ssim_enc = ssim_index(orig, encrypted)

    print("\n=== Original vs Encrypted (for security contrast) ===")
    print(f"MSE  : {mse_enc:.6f}")
    print(f"PSNR : {psnr_enc:.2f} dB")
    print(f"SSIM : {ssim_enc:.4f}")

    return {'input': input_path, 'encrypted': enc_path, 'decrypted': dec_path, 'K': K, 'encrypted_key': encrypted_key, 'pubkey': pubkey, 'privkey': privkey}

if len(sys.argv) > 1:
    image_file_name = sys.argv[1]
    ukuran_kunci = sys.argv[2]
else:
    raise Exception("Missing image file name")

# Run demo pipeline
outputs = demo_pipeline(image_file_name, int(ukuran_kunci))
outputs