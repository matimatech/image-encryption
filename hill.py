import imageio.v3 as iio
import numpy as np

class AdvHill:
    """
        Generate involutory matrix
        
        A = |A11    A12|
            |A21    A22|
        
        A22 = (n/2) * (n/2)
        A_11 = - A22
        A_12 = k(I - A11) % 256
        A_21 = A_21 * (1/k) % 256
        
        Paramater:
        n: size of matrix

        Return:
        A: matrix involutory
    """        
    def __init__(self, img, mod = 256):
        self.img = img
        self.mod = mod

    def _generate_key(self, n):
        # Algorithm 3.1.1. An Involutory Key Matrix
        A_22 = np.random.randint(256, size = (int(n/2),int(n/2)))          #Arbitrary Matrix, should be saved as Key also

        I = np.identity(int(n/2))

        A_11 = np.mod(-A_22, self.mod)
        k = 10 # NOTE: pilih k, suatu scalar

        A_12= np.mod((k * np.mod(I - A_11, self.mod)), self.mod)
        k = np.mod(np.power(k, 127), self.mod)
        A_21 = np.mod((I + A_11), self.mod)
        A_21 = np.mod(A_21 * k, self.mod)

        A1 = np.concatenate((A_11, A_12), axis = 1)
        A2 = np.concatenate((A_21, A_22), axis = 1)
        self.A = np.concatenate((A1,A2), axis = 0)

    def encrypt(self):
        """Encrypt function"""
        Enc1 = (np.matmul(self.A % self.mod, self.img[:,:,0] % self.mod)) % self.mod 
        Enc2 = (np.matmul(self.A % self.mod, self.img[:,:,1] % self.mod)) % self.mod 
        Enc3 = (np.matmul(self.A % self.mod, self.img[:,:,2] % self.mod)) % self.mod 

        Enc1 = np.resize(Enc1,(Enc1.shape[0],Enc1.shape[1],1))
        Enc2 = np.resize(Enc2,(Enc2.shape[0],Enc2.shape[1],1))
        Enc3 = np.resize(Enc3,(Enc3.shape[0],Enc3.shape[1],1))

        self.Enc = np.concatenate((Enc1, Enc2, Enc3), axis = 2).astype(np.uint8)                #Enc = A * image
        iio.imwrite(f'Encrypted.png', self.Enc)
        return self.Enc

    def decrypt(self, encrypted_img):
        """Decode function"""
        Enc = encrypted_img
        l = (self.A[-1][0] * self.mod + self.A[-1][1]).astype(np.uint8) # The length of the original image 
        w = (self.A[-1][2] * self.mod + self.A[-1][3]).astype(np.uint8) # The width of the original image
        print(self.A)
        # self.A = self.A[0:-1]

        Enc = Enc.astype(np.uint16)

        Dec1 = (np.matmul(self.A % self.mod, Enc[:,:,0] % self.mod)) % self.mod # R
        Dec2 = (np.matmul(self.A % self.mod, Enc[:,:,1] % self.mod)) % self.mod # G
        Dec3 = (np.matmul(self.A % self.mod, Enc[:,:,2] % self.mod)) % self.mod # B

        Dec1 = np.resize(Dec1, (Dec1.shape[0], Dec1.shape[1], 1))
        Dec2 = np.resize(Dec2, (Dec2.shape[0], Dec2.shape[1], 1))
        Dec3 = np.resize(Dec3, (Dec3.shape[0], Dec3.shape[1], 1))
        print(f"Dec1 {Dec1.shape}, Dec2 {Dec2.shape} Dec3 {Dec3.shape}")
        
        Dec = np.concatenate((Dec1, Dec2, Dec3), axis = 2)                #Dec = A * Enc
        print(f"DEC = {Dec}")
        
        Final = Dec.astype(np.uint8)                                      #Returning Dimensions to the real image

        print(f"Final = {Final.shape}")
        iio.imwrite('Decrypted.png',Final)
        return Final