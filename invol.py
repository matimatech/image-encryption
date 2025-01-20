import random
import numpy as np
from numpy.linalg import eig
import imageio.v3 as iio

#TODO: Generate A_22 ==> (n-1)x(n-1)



def generate_involutory_matrix(n):
    """
    Generate involutory matrix
    Paramater:
    n: size of matrix

    Return:
    A: matrix involutory
    """

    A_22 = np.random.randint(256, size = (int(n/2),int(n/2)))          #Arbitrary Matrix, should be saved as Key also
    
    MOD = 256
    k = 23 

    I = np.identity(int(n/2))
    A_11 = np.mod(-A_22,MOD)

    A_12= np.mod((k * np.mod(I - A_11, MOD)), MOD)
    k = np.mod(np.power(k, 127), MOD)
    A_21 = np.mod((I + A_11), MOD)
    A_21 = np.mod(A_21 * k, MOD)

    A1 = np.concatenate((A_11, A_12), axis = 1)
    A2 = np.concatenate((A_21, A_22), axis = 1)

    A = np.concatenate((A1,A2), axis = 0)
    
    Test = np.mod(np.matmul(np.mod(A, MOD), np.mod(A, MOD)), MOD)
    # Saving key as an image
    key = np.zeros((n + 1, n))
    key[:n, :n] += A
    # Adding the dimension of the original image within the key
    # Elements of the matrix should be below 256
    Mod = 256
    key[-1][0] = int(l / Mod)
    key[-1][1] = l % Mod
    key[-1][2] = int(w / Mod)
    key[-1][3] = w % Mod
    key = key.astype(np.uint8)


    iio.imwrite("Key.png", key)
    return A, Test

# print(A)

img = iio.imread("docs/images/lena.png")
print(img.shape)
l = img.shape[0]
w = img.shape[1]
n = max(l,w)
if n%2:
    n = n + 1
img2 = np.zeros((n,n,3))
img2[:l,:w,:] += img    
A, _ = generate_involutory_matrix(n)

# Saving key as an image
key = np.zeros((n + 1, n))
key[:n, :n] += A
# Adding the dimension of the original image within the key
# Elements of the matrix should be below 256
Mod = 256
key[-1][0] = int(l / Mod)
key[-1][1] = l % Mod
key[-1][2] = int(w / Mod)
key[-1][3] = w % Mod
key = key.astype(np.uint8)


iio.imwrite("Key.png", key)

#-------------Encrypting-------------
Enc1 = (np.matmul(A % Mod,img2[:,:,0] % Mod)) % Mod
Enc2 = (np.matmul(A % Mod,img2[:,:,1] % Mod)) % Mod
Enc3 = (np.matmul(A % Mod,img2[:,:,2] % Mod)) % Mod

Enc1 = np.resize(Enc1,(Enc1.shape[0],Enc1.shape[1],1))
Enc2 = np.resize(Enc2,(Enc2.shape[0],Enc2.shape[1],1))
Enc3 = np.resize(Enc3,(Enc3.shape[0],Enc3.shape[1],1))
Enc = np.concatenate((Enc1,Enc2,Enc3), axis = 2)                #Enc = A * image
Enc = Enc.astype(np.uint8)
iio.imwrite('Encrypted.png', Enc)

#-------------Decrypting-------------
Enc = iio.imread('Encrypted.png')                           #Reading Encrypted Image to Decrypt
# Loading the key
A = iio.imread('Key.png')
l = A[-1][0] * Mod + A[-1][1] # The length of the original image 
w = A[-1][2] * Mod + A[-1][3] # The width of the original image
A = A[0:-1]

Dec1 = (np.matmul(A % Mod,Enc[:,:,0] % Mod)) % Mod
Dec2 = (np.matmul(A % Mod,Enc[:,:,1] % Mod)) % Mod
Dec3 = (np.matmul(A % Mod,Enc[:,:,2] % Mod)) % Mod

Dec1 = np.resize(Dec1,(Dec1.shape[0],Dec1.shape[1],1))
Dec2 = np.resize(Dec2,(Dec2.shape[0],Dec2.shape[1],1))
Dec3 = np.resize(Dec3,(Dec3.shape[0],Dec3.shape[1],1))
Dec = np.concatenate((Dec1,Dec2,Dec3), axis = 2)                #Dec = A * Enc

Final = Dec[:l,:w,:]                                            #Returning Dimensions to the real image

iio.imwrite('Decrypted.png',Final)