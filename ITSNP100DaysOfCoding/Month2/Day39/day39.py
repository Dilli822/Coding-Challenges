import numpy as np
import matplotlib.pyplot as plt

# array creation using array method
arr = np.array([1, 2, 3, 4, 5])

print(arr)

print(type(arr))


# [1 2 3 4 5]
# <class 'numpy.ndarray'>


# we can pass list, tuple ,objects inside the array() method

arrList = np.array((1, 2, 3, 4, 5))

print(arrList)
print(type(arrList))

# [1 2 3 4 5]
# <class 'numpy.ndarray'>
# [21 26 32 40 15]
# <class 'numpy.ndarray'>


# Draw a line in a diagram from position (0, 1) to position (10, 15):
# x1 = 0 and x2 = 10 and y1 = 1 and y2 = 15

xpoints = np.array([0,1])
ypoints = np.array([10,15])
plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')
plt.plot(xpoints, ypoints)
plt.show()

x_points = np.array([1, 5])
y_points = np.array([3, 10])

plt.plot(x_points, y_points, "o")
plt.show()