import numpy as np


def generate_involutory_matrix(n):
    a11 = np.mod(np.random.random_integers(0, 100, (int(n / 2), int(n / 2))), 31)
    a12 = np.mod(-a11, 31)

    return a11, a12

    # while true:
    #     # generate a random integer matrix
    #     matrix = np.random.randint(0, 10, (n, n))
    #
    #     # check if the matrix is involutory: matrix * matrix = identity matrix (mod 26 for hill cipher)
    #     if np.array_equal(np.mod(np.dot(matrix, matrix), 26), np.identity(n)):
    #         return matrix


a11, a12 = generate_involutory_matrix(81)
print(f"a11 {a11}")
print(f"a11 shape {a11.shape}")
print(f"a12 {a12}")
