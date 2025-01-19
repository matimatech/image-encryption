import random
import numpy as np
from numpy.linalg import eig


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

    return A, Test

A, _ = generate_involutory_matrix(6)
print(A)