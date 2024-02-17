import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# Creating a TensorFlow constant with the numeric value 45
rank1 = tf.constant(45)
print(rank1)
# Creating a NumPy array
array = np.array([23, 2, 45, 67])
# Adding array elements with indexing
print(array[0] + array[2])

# Accessing elements of a 2D NumPy array
array2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
# Accessing the element at the first row, first column
print(array2[0, 0])
# second row and third column
print(array2[1, 2])
#accessing 3D array
array3 = np.array([ [ [1,2,3,4], [5,6,7,8], [9, 10, 11, 12] ]])
# accessing second element of the second array of the first array
print(array3[0, 1, 1])

# negative indexing
print(array[-1])
print("array negative indexing for 2d array -->", array2[0, -1])
print("array negative indexing for 3d array last element of the last array -->", array3[0, 2, -1])
y_points = [-4, -9, -16, 4, 9, 16]
x_points = [-2, -3, -4, 2, 3, 4]
plt.xlabel("X axis label")
plt.ylabel("Y axis label")
plt.plot(x_points, y_points, marker = 'o')
plt.show()

y_values = np.linspace(-5, 5, 100)  # Generate 100 values between -5 and 5
# Calculate corresponding values for x
x_values = y_values ** 2
plt.plot(x_values, y_values, marker='*')
plt.show()
plt.title('Plot of x = y^2')
plt.legend()

#fmt in matplot is a shortcut string notation to specify the marker
# marker|line|color
ypoints = [1,3,5,7,11]
plt.plot(y_points, 'o:r')
plt.title("Fmt marker ")
plt.show()
# # marker size
plt.plot(y_points, 'o:g', ms='25')
plt.show()
# marker color
plt.plot(y_values, marker='o', ms="20", mec="r")
plt.show()