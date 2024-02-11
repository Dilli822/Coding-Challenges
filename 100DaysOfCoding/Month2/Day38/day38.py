
# https://docs.google.com/document/d/12XKx4-R0_YSgdCQT0q3y8LJn87hWVHlFkWOffCIVdJQ/edit?usp=sharing
# install numpy using cmd pip install numpy
# install matplotlib   using cmd pip install matplotlib

# numpy = numerical python used for working with array in python

# Matplotlib is open source, a low level graph plotting library 
# in python that serves as a visualization utility.

# pyplot provides a MATLAB-like interface for creating a variety 
# of plots and visualizations. It allows users to create figures, add
# axes to the figures, plot data, customize the appearance 
# of plots, add labels and legends, and more.
import matplotlib
# importing the pyplot as plt pyplot
import matplotlib.pyplot as plt
import numpy as np

print(matplotlib.__version__)

# using numpy for array creation which serves as x and y points
xpoints = np.array([0, 10])
ypoints = np.array([0, 340])

# ploting the points
plt.plot(xpoints, ypoints)
plt.show()