

"""
MY Docs Link:
https://docs.google.com/document/d/1g3X082nmluU-jMGmZotAANUv6vPIDBz5IgZbT3gOIWQ/edit

* Sample Size in CNN
 This isnot really the best term to describe this but each convolutional layer is
 going to examine n x m blocks of pixels in each image.Type we will consider 3x3 or 5x5 
 blocks. In the example above we use a 3 x 3 "sample size". This size be the same as
 the size of our filter.
 
 Our layers work by sliding these filters of n x m pixels over every possible position
 in our image and populating a new feature map/response map indicating whether the
 filter is present at each location.
 
 Response Map -> Quantifying the response of the filter's pattern at different locations.
 
 Dense neural networks:
 Dense neural networks analyze input on a global scale and recognize patterns in specific areas.
 Dense neural network do not work well for image classification or object detection.
 It also analyze input globally and extract features from specific areas.
 
 Convolutional neural networks:
 Convolutional neural networks scan through the entire input a little at a time and learn local patterns.
 It returns a feature map that quantifies the presence of filters at a specific locations.
 and that filters advantages of it, we slide it in entire image and if this feature or filter
 is present in anywhere image then we will know the pattern in the image.

"""


"""

DRAWING BOARD - HOW CONVOLUTIONAL NETWORKS WORKS?
https://miro.com/app/board/uXjVNhgCCxA=/?share_link_id=982196115426


"""

"""
# NumPy ufuncs - universal functions that operates on ndarray object
# ufuncs takes three arguments where dtype and output
# vectorization is used in ufuncs which are more faster than iterating over elements
 what is vectorization ?
 it is a method of converting iterative statements into a vector based operation since
 all modern cpus are designed to support it.
 
 lets do an example by iterating over the two list and sum up them
"""

import numpy as np
from numpy import random
import matplotlib.pyplot as plt

x = [0,2,4,6,8]
y = [1,3,5,7,9]
z = []

# ---- SO THE WHOLE POINT IS .add() is ufunc or not
for i, j in zip(x,y):
    z.append(i + j)

print("adding two lists simulatneousl -->", z)
# instead of loop lets use add function directly

z = np.add(x,y)
print("using add method--> ", z)

# creating ufunc
def customadd(x, y):
  return x+y

myadd = np.frompyfunc(customadd, 2, 1)
print(myadd([1, 2, 3, 4], [5, 6, 7, 8]))
# output is --> <class 'numpy.ufunc'>
print(type(np.add))

print(type(np.concatenate))
# output is --> <class 'numpy._ArrayFunctionDispatcher'>
# print(type(np.blahblah))
#     raise AttributeError("module {!r} has no attribute "
# AttributeError: module 'numpy' has no attribute 'blahblah'

if type(np.add) == np.ufunc:
    print("custom add is ufunc")
else:
    print("custom add is not ufunc")
    
# Drawing 2 plots on top of each other:
plt.subplot(2, 1, 1)
plt.title("2 ROW 1 COLUMNS INDEXED 1")
plt.plot(x, y, marker="*", color="m", ms=10)

plt.subplot(2, 1, 2)
plt.title("2 ROW 1 COLUMNS INDEXED 2")
plt.plot(x, z, marker='.', color='c', ms=20)

plt.suptitle("SUPER TITLE")
plt.show()

# for 6 plots
# Data for plots
x_values = np.array([0, 1, 2, 3])

# First set of y values
y1_values = np.array([10, 20, 30, 40])

# Second set of y values
y2_values = np.array([3, 8, 1, 10])

# Creating subplots
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
# Plot 1
axs[0, 0].plot(x_values, y1_values, color='blue', linestyle='-', marker='o')
axs[0, 0].set_title('Plot 1')

# Plot 2
axs[0, 1].plot(x_values, y2_values, color='red', linestyle='--', marker='s')
axs[0, 1].set_title('Plot 2')

# Plot 3
axs[0, 2].plot(x_values, y1_values, color='green', linestyle='-.', marker='x')
axs[0, 2].set_title('Plot 3')

# Plot 4
axs[1, 0].plot(x_values, y2_values, color='orange', linestyle=':', marker='d')
axs[1, 0].set_title('Plot 4')

# Plot 5
axs[1, 1].plot(x_values, y1_values, color='purple', linestyle='-', marker='^')
axs[1, 1].set_title('Plot 5')

# Plot 6
axs[1, 2].plot(x_values, y2_values, color='brown', linestyle='--', marker='v')
axs[1, 2].set_title('Plot 6')

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()