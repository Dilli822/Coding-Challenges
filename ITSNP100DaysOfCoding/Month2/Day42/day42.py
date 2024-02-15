import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 

# tensorflow allows all the basic ops 
# addition, multiplication, subtraction and matrix multiplication 

a = tf.constant([ [1, 2], [3, 4] ])
b = tf.constant([ [2, 3], [4, 5] ])
a @ b
print(a @ b)

complex_number = tf.constant(4.0 + 5.0j)

print(complex_number)  # tf.Tensor((4+5j), shape=(), dtype=complex128)


# getting real and imaginary parts 
real_parts = tf.math.real(complex_number)
imaginary_parts = tf.math.imag(complex_number)

print("real part", real_parts) # real part tf.Tensor(4.0, shape=(), dtype=float64)
print("imaginary part", imaginary_parts)   # imaginary part tf.Tensor(5.0, shape=(), dtype=float64)

# The base tf.Tensor class requires tensors to be "rectangular"---that is, along each axis, every element is the same size. 
# However, there are specialized types of tensors that can handle different shapes:
# Ragged tensors
# Sparse tensors 


# n-dim and n-dmin higher dimensional arrays
narray = np.array([1, 2, 3, 4], ndmin = 10)
print(narray)
print("Dimension of an array is ", narray.ndim)