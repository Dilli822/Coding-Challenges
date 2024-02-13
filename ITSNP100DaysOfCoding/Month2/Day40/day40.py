import matplotlib.pyplot as plt
import numpy as np

# xpoints = np.array([0, 10])
# ypoints = np.array([2, 10])
# plt.xlabel("Age as X-axis label ")
# plt.ylabel(" BlooD Pressure as y-axis label ")

# plt.plot(xpoints, ypoints, "o")


# Multiple Points
# we can plot as many number as but both array should have equal number of elements in them
# x_points = np.array([0, 2, 4, 6, 8, 10, 12])
# y_points = np.array([1, 3, 5, 7, 9, 13, 17])

# plt.plot(y_points)
# plt.show()


xpoints = np.array([1, 2, 6, 8])
ypoints = np.array([3, 8, 1, 10])

plt.xlabel("X-axis points")
plt.ylabel("Y-axis points")
plt.title("Multiple Points")
plt.plot(xpoints, ypoints)
plt.show()


# Default X-Points - if we donot set the x-axis points then it will be by default 
# 0,1,2,.... depends on the length of the y-points


ypt = np.array([3, 8, 1, 10, 5, 7])

plt.plot(ypt)
plt.xlabel("X-axis points")
plt.ylabel("Y-axis points")
plt.title("Default X-points Points")
plt.show()


# 0-D array
ZeroD = np.array(45)
print(ZeroD)

# 1-D Array
OneD = np.array([1,2,3,4,5])
print(OneD)

# 2-D Array
TwoD = np.array( [ [1,2,3,4,5], [7,8,9,10,11] ])
print(TwoD)


# 3-D Array
ThreeD = np.array([ [1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15] ])
print(ThreeD)


# CHECKING DIMENSION USING ndim() method

a = np.array(42)
b = np.array([1, 2, 3, 4, 5])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)


# output
# 45
# [1 2 3 4 5]
# [[ 1  2  3  4  5]
#  [ 7  8  9 10 11]]
# [[ 1  2  3  4  5]
#  [ 6  7  8  9 10]
#  [11 12 13 14 15]]
# 0
# 1
# 2
# 3