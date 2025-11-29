import numpy as np
import math
import random

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
            
    # Langkah 5: Hitung k_inv
    k_inv = mod_inverse(k, mod)
    
    # Langkah 6: Hitung A12
    A12 = (k * (I - A11)) % mod
    
    # Langkah 7: Hitung A21
    A21 = (k_inv * (I + A11)) % mod
    
    # Langkah 8: Bentuk Matriks K
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

# --- Eksekusi ---

# 1. Menjalankan contoh manual 2x2 dari diskusi kita
K_manual = generate_involutory_key_manual(A22_scalar=5, k=3, mod=256)
validate_key(K_manual, 256)

# 2. Membuat kunci acak 4x4
# (Catatan: n=4, sehingga sub_size=2. A11, A12, A21, A22 akan menjadi matriks 2x2)
K_random_4x4 = generate_involutory_key_random(n=4, mod=256)
validate_key(K_random_4x4, 256)