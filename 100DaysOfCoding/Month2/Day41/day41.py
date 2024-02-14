import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define a TensorFlow constant tensor
tensor_a = tf.constant([[1, 2], [3, 4]])

# Evaluate the tensor
result = tensor_a.numpy()

print("Evaluated tensor:")
print(result)

# combination of scalar , vector and matrix = tensor
# Scalar -rank 0 no axes
# Vector - rank 1 have 1 axes
# Matrix - rank 2  have 2 axes

rank_0 = tf.constant(7)
print(rank_0)

# tensor is like a list of values. 
rank_1 = tf.constant([2.00, 4.00, 6.00])
print(rank_1)

# tensor rank 2 is matrix [row, col]
rank_2 = tf.constant([ [3, 5], [7, 8] ])
print(rank_2)


# besides there are n number of axes or we called dimnesions of tensor

n_rank = tf.constant( [ [ [2,4,5], [6,7,8], [9,2,1] ]])
print(n_rank)

rank_3_tensor = tf.constant([
  [[0, 1, 2, 3, 4],
   [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24],
   [25, 26, 27, 28, 29]],])

print(rank_3_tensor)

# Define the rank 4 tensor
rank_4_tensor = tf.constant([
  
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10]
            ],
            [
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20]
            ],
            [
                [21, 22, 23, 24, 25],
                [26, 27, 28, 29, 30]
            ],
            [
                [31, 32, 33, 34, 35],
                [36, 37, 38, 39, 40]
            ]
])

# Evaluate the tensor using eager execution
print(rank_4_tensor)
print(rank_4_tensor.numpy())

# output
#dilli@Dillis-Air Day41 % python3 day41.py
# Evaluated tensor:
# [[1 2]
#  [3 4]]
# tf.Tensor(7, shape=(), dtype=int32)
# tf.Tensor([2. 4. 6.], shape=(3,), dtype=float32)
# tf.Tensor(
# [[3 5]
#  [7 8]], shape=(2, 2), dtype=int32)
# tf.Tensor(
# [[[2 4 5]
#   [6 7 8]
#   [9 2 1]]], shape=(1, 3, 3), dtype=int32)
# tf.Tensor(
# [[[ 0  1  2  3  4]
#   [ 5  6  7  8  9]]

#  [[10 11 12 13 14]
#   [15 16 17 18 19]]

#  [[20 21 22 23 24]
#   [25 26 27 28 29]]], shape=(3, 2, 5), dtype=int32)
# tf.Tensor(
# [[[ 1  2  3  4  5]
#   [ 6  7  8  9 10]]

#  [[11 12 13 14 15]
#   [16 17 18 19 20]]

#  [[21 22 23 24 25]
#   [26 27 28 29 30]]

#  [[31 32 33 34 35]
#   [36 37 38 39 40]]], shape=(4, 2, 5), dtype=int32)
# [[[ 1  2  3  4  5]
#   [ 6  7  8  9 10]]

#  [[11 12 13 14 15]
#   [16 17 18 19 20]]

#  [[21 22 23 24 25]
#   [26 27 28 29 30]]

#  [[31 32 33 34 35]
#   [36 37 38 39 40]]]
