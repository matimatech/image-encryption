import os.path
import pickle

import numpy as np
from numpy.linalg import det, inv

MOD = 256
class Hill:
    def __init__(self, img, file_name, key_path=None):
        self.img = img
        self.file_name = file_name


    def encrypt(self, data):
        """Encrypt function"""
        Enc1 = (np.matmul(A % MOD,img[:,:,0] % MOD)) % MOD
        Enc2 = (np.matmul(A % MOD,img[:,:,1] % MOD)) % MOD
        Enc3 = (np.matmul(A % MOD,img[:,:,2] % MOD)) % MOD

        Enc1 = np.resize(Enc1,(Enc1.shape[0],Enc1.shape[1],1))
        Enc2 = np.resize(Enc2,(Enc2.shape[0],Enc2.shape[1],1))
        Enc3 = np.resize(Enc3,(Enc3.shape[0],Enc3.shape[1],1))
        Enc = np.concatenate((Enc1,Enc2,Enc3), axis = 2)                #Enc = A * image
        Enc = Enc.astype(np.uint8)
        iio.imwrite('Encrypted.png', Enc)

    def decode(self, data):
        """Decode function"""
      