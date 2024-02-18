import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf

#Indexing rule is same as the way we do in the python list or string using numpy
# indexing starts from 0 and negative means from the backend whereas adding : will slice
# start: stop : step
rank1 = tf.constant([1,2,3,4,5])
print(rank1)
print(rank1.numpy())

# accessing elements using indexing along with numpy
# indexing with scalar constant removes the axes
print("Example of single axes indexing ")
print("The first element is --> ", rank1[0].numpy)
print("The second element is --> ", rank1[-1].numpy)
print("Slicing techique--> ", rank1[3].numpy)

print("Indexing with slices : keep the axis.")
example1 = tf.constant( [ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
print("E√erything ", example1[:].numpy())
print("Rreversed ", example1[::-1].numpy())
print("Escaping each element ", example1[::1].numpy())
print("from mid half ", example1[:5].numpy())

# slicing in numpy
array1 = np.array([2,4,6,8, 45, 47, 89])
print(array1[0])
print("starts from index 1-->", array1[:1])
print("starts from last index -->", array1[:-1])
print("starts from zero index to using 2 step to the 4th index ", array1[1:3:2])

print("using mec and mfc arguments to color entire marker ")
plt.plot(array1, marker="o", mec="g", mfc="r")
plt.legend("Using mec and mfc")
plt.show()

print("WE CAN ALSO USE HEX CODE")
plt.plot(array1, marker=".", ms="20", mec = '#4CAF50', mfc = '#4CAF50')
plt.title("with custom mec and mfc ")
plt.show()
