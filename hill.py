import os.path
import pickle

import imageio.v3 as iio
import numpy as np


class AdvHill:
    def __init__(self, img, file_name, mod = 256):
        self.img = img
        self.file_name = file_name
        self.mod = mod

    def _generate_key(self, n):
        """
        Generate involutory matrix
        Paramater:
        n: size of matrix

        Return:
        A: matrix involutory
        """

        A_22 = np.random.randint(256, size = (int(n/2),int(n/2)))          #Arbitrary Matrix, should be saved as Key also
        k = 23 

        I = np.identity(int(n/2))
        A_11 = np.mod(-A_22,self.mod)

        A_12= np.mod((k * np.mod(I - A_11, self.mod)), self.mod)
        k = np.mod(np.power(k, 127), self.mod)
        A_21 = np.mod((I + A_11), self.mod)
        A_21 = np.mod(A_21 * k, self.mod)

        A1 = np.concatenate((A_11, A_12), axis = 1)
        A2 = np.concatenate((A_21, A_22), axis = 1)
        self.key = np.concatenate((A1,A2), axis = 0)


    def encrypt(self, n):
        """Encrypt function"""
        self._generate_key(n)
        print(self.file_name)
        Enc1 = (np.matmul(self.key % self.mod, self.img[:,:,0] % self.mod)) % self.mod 
        Enc2 = (np.matmul(self.key % self.mod, self.img[:,:,1] % self.mod)) % self.mod 
        Enc3 = (np.matmul(self.key % self.mod, self.img[:,:,2] % self.mod)) % self.mod 

        Enc1 = np.resize(Enc1,(Enc1.shape[0],Enc1.shape[1],1))
        Enc2 = np.resize(Enc2,(Enc2.shape[0],Enc2.shape[1],1))
        Enc3 = np.resize(Enc3,(Enc3.shape[0],Enc3.shape[1],1))
        Enc = np.concatenate((Enc1,Enc2,Enc3), axis = 2)                #Enc = A * image
        Enc = Enc.astype(np.uint8)
        iio.imwrite(f'Encrypted.png', Enc)
        return Enc

    def decrypt(self, encrypted_img):
        """Decode function"""
        Enc = encrypted_img

        l = self.key[-1][0] * self.mod + self.key[-1][1] # The length of the original image 
        w = self.key[-1][2] * self.mod + self.key[-1][3] # The width of the original image
        Enc = Enc.astype(np.uint16)
        Dec1 = (np.matmul(self.key % self.mod, Enc[:,:,0] % self.mod)) % self.mod # R
        Dec2 = (np.matmul(self.key % self.mod, Enc[:,:,1] % self.mod)) % self.mod # G
        Dec3 = (np.matmul(self.key % self.mod, Enc[:,:,2] % self.mod)) % self.mod # B
        print(f"Dec1 = {Dec1.shape} | Dec2 = {Dec2.shape} | Dec3 {Dec3.shape}")

        Dec1 = np.resize(Dec1,(Dec1.shape[0],Dec1.shape[1],1))
        Dec2 = np.resize(Dec2,(Dec2.shape[0],Dec2.shape[1],1))
        Dec3 = np.resize(Dec3,(Dec3.shape[0],Dec3.shape[1],1))
        Dec = np.concatenate((Dec1,Dec2,Dec3), axis = 2)                #Dec = A * Enc
        Dec = Dec.astype(np.uint8)

        # Final = Dec[:l,:w,:]                                            #Returning Dimensions to the real image
        iio.imwrite('Decrypted.png',Dec)
        return Dec
