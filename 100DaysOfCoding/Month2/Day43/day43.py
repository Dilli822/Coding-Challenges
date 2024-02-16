import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# day 43 
# about shapes in tensor
# rank, dimension, size, shape

# rank = number of axes
# dimensions or axis 
# size = total no of items in the tensor
# shape = number of each element in the tensor

# tf.TensorShape has convinent properties

rank_4 = tf.zeros([3, 2, 4, 5])
print(rank_4)

print("Type of every element: ", rank_4.dtype)
print("Number of axes:", rank_4.ndim)
print("Shape of tensor:", rank_4.shape)
print("Elements along axis 0 of tensor:", rank_4.shape[0])
print("Elements along the last axis of tensor:" ,rank_4.shape[-1])
print("Total number of elements (3*2*4*5):  120", tf.size(rank_4).numpy())


# ndim and dtype and dshape are not the same since ndim donot return objects


# ragged tensor - a variable number of elements in the list along withte some axis
# for ragged tensor [4, None]
ragged_list = [
    [1,2,3,4],
    [23, 45],
    [12],
    [4,5,6]
]

print(ragged_list)
try:
    tensor = tf.constant(ragged_list) # ValueError: Can't convert non-rectangular Python sequence to Tensor.
except Exception as e:
    print(f"{type (e).__name__}: {e}")
# we need Instead create a tf.RaggedTensor using tf.ragged.constant:
# print(ragged.shape) # gives error 
ragged_tensor = tf.ragged.constant(ragged_list)
print(ragged_tensor)

# now we can access the properites
print(ragged_tensor.shape)

# array indexing in numpy
arr = np.array([12,34,56,78,0])
print("accessing first element-->", arr[0])
print("accessing the last element ", arr[-1])

x_points = np.linspace(-10, 10, 100)
y_points = x_points ** 2

plt.xlabel("plot of y = x^2'")
plt.ylabel("y -axis")
plt.plot(x_points, y_points)
plt.grid(True)
plt.show()