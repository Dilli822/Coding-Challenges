# Input layer
# Hidden Layer
# Output Layer
# Bias
# --- it is not input but a constant numeric value
# --- it gets connected to each hidden layers and output layer neurons
# --- it is trainable 
# --- whenever bias is connected to another layer then its weight is 1
# ---- bias never gets added or connected with each other
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
# Parameters
mu = 0  # Mean of the distribution
sigma = 1  # Standard deviation of the distribution
data_points = 1000  # Number of data points
# Generate data points
x = np.linspace(mu - 3*sigma, mu + 3*sigma, data_points)
print("X axis data points for bell shaped ", x)
y = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))
print("y axis data points for bell shaped ", y)
# Plotting
plt.plot(x, y, color='b')
plt.title('Bell-Shaped Curvy Data Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()
# data distrbuion - collection of lists that contain the duplicate numbers or lists too
# random distribution - a set of random numbers that follows the probability density function 
# using .choice method and  with random.choice
# setting the probablity for each elements of an array
data_distribution = random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(100))
print("data distribution ", data_distribution) # value of 9 never occurs probability is 0
# 2d array with 3 rows containing 5 values
data_distribution_2darray = random.choice([0, 2, 4, 6], p=[0, 0.2, 0.6, 0.2], size=(3, 5))
print("for 2d array --> ", data_distribution_2darray)
# Generate data points
x = np.linspace(0, 2*np.pi, 1000)  # Values from 0 to 2*pi
y = np.sin(x)
# Plotting
plt.plot(x, y, color='r')
plt.title('Sine Curve')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)
plt.show()
# Generate data points
x = np.linspace(0, 5, 1000)  # Values from 0 to 5
y = np.exp(-x)
# Plotting
plt.plot(x, y, color='g')
plt.title('Exponential Decay Curve')
plt.xlabel('x')
plt.ylabel('exp(-x)')
plt.grid(True)
plt.show()
# Generating 10 evenly spaced numbers between 0 and 1 (inclusive)
values = np.linspace(0, 1, 10)
print(values)
#[0.         0.11111111 0.22222222 0.33333333 0.44444444 0.55555556
# 0.66666667 0.77777778 0.88888889 1.        ]
x = 90
print(np.sin(x))