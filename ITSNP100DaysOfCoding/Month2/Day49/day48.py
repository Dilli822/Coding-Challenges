
import numpy as np
import matplotlib.pyplot as plt
# the shape of an array is nothing but the number of elements in each dimension
# we can reshape our array using reshape 
print("simple reshaping example with reshape")

arr = np.array([ 1, 2, 3, 4, 5, 6, 7, 8 ])
# newarr = arr.reshape(4,3)
#     newarr = arr.reshape(4,3)
#              ^^^^^^^^^^^^^^^^
# ValueError: cannot reshape array of size 9 into shape (4,3)
newarr = arr.reshape(2, 4) # 2 x 4
newarr2 = arr.reshape(4, 2) # 4 x 2
print(newarr)
print(newarr2)
print("We reshape into any shape if only both are equal in shapes")
print("using base will return an original array even the reshaped array ")
arre = np.array( [ 1,2,3,4,5,6,7,8,9,10,11,12,13,14])
arre2 = arre.reshape(2, 1, 7)
print(arre2)
print( arre.reshape(2, 1, 7).base)

# unknown dimension
# we are allowed to have only one unknown dimension which is -1
# sometimes I may not to able to determine the number of dimension in the reshape method
ary = np.array([23, 45, 67, 8, 5, 6, 7, 8])
print("i donot know the no of array keeping -1 will auto reshape ", ary.reshape(2,2,-1))
# flatenning array very important 
# converting multiple array into a single array
rey = np.array( [ [1,2,3], [4,5,6] ])
print("before flattening", rey)
print("after flatening ", rey.reshape(-1))
x = np.array([1, 3, 5, 7, 9, 12])
y = np.array([2, 4, 6, 7, 12, 11])
# grid in matplot
plt.plot(x, y)
plt.grid()
plt.title("GRID IN MATPLOT")
plt.show()
#output
# dilli@Dillis-Air Day49 % python3 day48.py
# simple reshaping example with reshape
# [[1 2 3 4]
#  [5 6 7 8]]
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]
# We reshape into any shape if only both are equal in shapes
# using base will return an original array even the reshaped array 
# [[[ 1  2  3  4  5  6  7]]

#  [[ 8  9 10 11 12 13 14]]]
# [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14]
# i donot know the no of array keeping -1 will auto reshape  [[[23 45]
#   [67  8]]

#  [[ 5  6]
#   [ 7  8]]]
# before flattening [[1 2 3]
#  [4 5 6]]
# after flatening  [1 2 3 4 5 6]
