    # 1.Write a Python/R program to create three
    # vectors a,b,c with 3 integers. Combine the
    # three vectors to become a 3×3 matrix where
    # each column represents a vector. Print the
    # content of the matrix.

import numpy as np

a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.array([7,8,9])

matrix = np.column_stack((a,b,c))
matrix1 = np.vstack((a,b,c))
print(matrix)
print(matrix1)