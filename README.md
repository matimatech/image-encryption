# Image Encryption
---
Projek ini merupakan implementasi dari algoritma kriptografi Advance Hill Cipher (AdvHill) dengan kunci terenkripsi ElGamal untuk enkripsi citra digital berwarna.

### Todo:
- [x] Membangkit matriks kunci self invertible
- [ ] Mengimplementasikan algoritma Advance Hill Cipher dengan mengacukan pada paper (Acharya et al)
- [ ] Mengimplementasikan algoritma ElGamal
- [ ] Mengecek performa dari algoritma enkripsi
- [ ] Membuat web aplikasi dengan menggunakan streamlit 

### Hasil:
### 
Original Image
<img src="docs/images/face.png" alt="isolated" width="200"/>

Encoded Image
<img src="docs/images/face-encoded.png" alt="isolated" width="200"/>


## Menjalankan Projek
### Get the code
```shell
$ git clone https://github.com/matimatech/image-encryption.git
$ cd image-encryption
```

### Create virtual environment
```shell
$ python3 -m venv venv
$ source venv/bin/activate
```

### Install dependencies
```shell
$ pip install -r requirements.txt
```

### Encode an image
```shell
$ python3 main.py image.jpg
```

### Source:
