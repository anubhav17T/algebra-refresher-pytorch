import numpy as np

x = np.array([1,2,3])
print(x.shape)
print(x)
print(len(x))
print(type(x))

x_transpose = x.T
print(x_transpose.shape)
print(x_transpose)



#matrix style

x_matrix = np.array([[1,2,3]])
print(x_matrix.shape)
print(type(x_matrix))
x_matrix_transpose = x_matrix.T
print(x_matrix_transpose)
print(x_matrix_transpose.shape)



print("-------- VECTORS IN PYTROCH ------")
import torch

x_vector = torch.tensor(data=[1,2,3],device='cpu')
print(x_vector.shape)
x_vector_transpose = x_vector.transpose()
print(x_vector_transpose)