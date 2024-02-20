
import numpy as np
import matplotlib.pyplot as plt

# concept of copy and view in numpy
# copy will just copy the original array into a new array whereas view will only view the original array

# see the difference between copy and view
array = np.array([ 45, 56, 98, 100])
new_array = array.copy()
new_array[-1] = 0  # assiging new value to the array
print("original array --> ", array)
print("new array with copy --->", new_array)

print("lets use only view instead of copy and change the value")
array2 = np.array([1, 2, 3, 4, 5])
new_array2 = array2.view()
new_array2[0] = 200
print("original array using only view not using copy -->", array2)
print("new array2 after value changed --> ",new_array2 )
print("Both value changed")

# since we maybe using view and copy method simultaneously so we need to check either copy owns the data
# copy copies the data so it owns the data but view only views the data
# before that we must remember that every numpy array has base attribute
# and base returns None if array owns the data
# if not returned None that means data is not copied and refers to the original array

arr = np.array([ "hello", "world", "I'm ", "Dilli"])
x = arr.copy()
y = arr.view()

print("checking if array x owns the data ", x.base) # owns the data
print("checking if array y owns the data ", y.base) # donot owns so only point to the original array data

# Numpy has shape attribute to know the number of elements in the array that determine the dimension of an array
print("2-D Numpy Array ")
array2d = np.array([ [1, 2, 3, 4], [23.36, 56.44, 67.77, 667.777 ]])
print("shape of 2d numpy array ", array2d.shape)
# it is 2 dimension array with 4 elements inside it return tuples (2, 4)

mutliarr = np.array( [1, 1, 1, 1], ndmin =5)
print("multi array ", mutliarr)
print("5 dimension array with each dimension having 1's ")
print(mutliarr.shape)

# using fontdict for styling the labels and titles 
x = [1, 3, 5, 7, 9]
y = [2, 4, 6, 8, 10]
# # dictionary for fonts 
font1 = { 'family': 'cursive', 'color': 'red', 'size': 20}
font2 = { 'family': 'serif', 'color': 'blue', 'size': 10}

plt.plot(x,y)
plt.xlabel("x-axis age", fontdict=font2)
plt.ylabel("y-axis height")
plt.title("Relationship between age and height ", fontdict=font1)
plt.show()

# usig loc parameter to locate either left, right or mid

plt.plot(x,y, marker='o')
plt.xlabel("x-axis age", fontdict=font1)
plt.ylabel("y-axis height ", fontdict=font2)
plt.title("Relationship between age and height ", loc='left')
plt.show()

# plt.plot(x,y, marker='o')
# plt.xlabel("x-axis age", fontdict=font1)
# plt.ylabel("y-axis height ", ontdict=font2)
# plt.title("Relationship between age and height ", loc='right')
# plt.show()