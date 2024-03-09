# ----------- Activation Functions -----------
# Rectified Linear unit.(Relu)
# it makes any numeric values less than 0 or -ve values as 0 and for +ve it is positive values
# Tanh(Hyberbolic Tanget)
# more positive closer to +ve and -ve closer to -ve just a normal graphs but ranges from -y to +x

# Sigmoid Function
# this function is to remove non linearity and is shaped-curve and any positive numbers closer 
# to 1 and any negative numbers closer 0


# --------- HOW TO USE THEM IN MODEL? ---------


# N1 = Activation function is applied to the output of that particular connected neurons 
# that means for n1 we apply activation function before sending its n1 output to another neuron m1
# WE CAN USE ANY ACTIVATION FUNCTION BASE ON THE REQUIREMENTS 

# FOR O TO 1 RANGE WE USE SIGMOID FUNCTION, SQUISHES VALUES BETWEEN O AND 1


# WHY TO USE ACTIVATION FUNCTION IN NEURAL INTERMEDIATE NETWORK LAYER?
# - TO INTROUDCE THE COMPLEXITY IN INPUT, WEIGHTTS AND BIAS
# - activation function is a higher dimensional function that spreads the 
#    linear or clustered points in a single dataset, spreading the dataset
#     will give us pattern, characteristics and some features of the data
#     moves in n dimensional data moves to 1D, 2D Or 3D data dimension
#     for example n dimension of shapes like SQUARE, CUBE

#  square provides less details then 
#  VOLUME/CUBE shaped 
# OUR AIM IS TO MOVE FRO AND BACK IN HIGHER AND LOWER DIMENSION TO 
# EXTRACT THE INFORMATION FROM THE DATA LIKE IN SQUARE WE CAN GET EITHER LENGTH
# OR BREADTH BUT IN CUBE WE CAN GET MORE DETAILS LENGTH, HEIGHT AND BREADTH
# in here for us is a matrix, scalar or event tensor of n dimension

# THAT EVENTUALLY LEADS TO BETTER PREDICTIONS 


#--------- LOSS FUNCTION --------
# 1. IT CALCULATE HOW FAR AWAY OUTPUT WAS FROM EXPECTED OUTPUT?
# suppose expected output was 1 but we get 0.2 and then how far it is from
# the expected and obtained output
# THAT MEANS WE CAN COMPARE HOW BAD OR GOOD IS NETWORK OUTPUT?
# BIASNESS CAN BE EVALUATED FROM LOSS FUNCTION
# HIGH LOSS MEANS VERY BAD NETWORK
# LOW LOSS MEANS NOT VERY BAD NETWORK
# AND IF HIGH LOSS CHANGE THE WEIGHT AND BIAS DRASCTICALLY AND GUIDE IT TO DIFFERENT OUTPUT OR PATH
# AND IF GOOD THEN THATS OKAY
# SINCE CHANGING OR TWISTING SMALL CHANGES IN THE NETWORK NUMERIC INPUT VALUES 
# CAN BRING VAST DIFFERENCE IN THE OUTOUT AND LOSS VALUE 

# install pip install seaborn
# Seaborn is a library that uses Matplotlib underneath to plot graphs. It will be used to visualize random distributions.
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Define the activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)
# Define the range of input values
x = np.linspace(-5, 5, 100)
# Compute the output of each activation function for the given range
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
# Plotting sigmoid function
plt.figure(figsize=(8, 6))
plt.plot(x, y_sigmoid, label='Sigmoid', color='blue')
plt.title('Sigmoid Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.legend()
plt.show()

# Plotting tanh function
plt.figure(figsize=(8, 6))
plt.plot(x, y_tanh, label='Tanh', color='red')
plt.title('Tanh Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.legend()
plt.show()

# Plotting ReLU function
plt.figure(figsize=(8, 6))
plt.plot(x, y_relu, label='ReLU', color='green')
plt.title('ReLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.legend()
plt.show()
# distplot means distrubution plot 
# takes an array inputs and curve corresponding to the distribution
sns.displot([1,3,5,7,9,11])
plt.show()
sns.distplot([0, 1, 2, 3, 4, 5], hist=False)
plt.show()


# Define the range of input values
x = np.linspace(-5, 5, 100)

# Compute the output of each activation function for the given range
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(x, y_sigmoid, label='Sigmoid', color='blue')
plt.plot(x, y_tanh, label='Tanh', color='red')
plt.plot(x, y_relu, label='ReLU', color='green')

plt.title('Activation Functions')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.legend()
plt.show()