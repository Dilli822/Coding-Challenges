
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import random

"""
COST OR LOSS FUNCTION
 THREE TYPES OF COST FUNCTION
1. MEAN ABSOLUTE
2. MEAN SQUARED
3. Hinge loss

Gradient Descent
Since our networks parameters are weights and
bias by just changing the bias and weights
we can make either the network better or worst

and the task of finding out that is done by 
LOSS Function
it determine how good or worse is our network

Based on that we can determine move the network
to change the worst scenario.

Link: https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html/
https://chat.openai.com/share/0d9c4dee-d141-44f6-9eec-69c8825131fd

FROM THE GRADIENT DESCENT FIGURE
Higher Dimension --> Higher Space , spread of dataset
Decrease the loss value to get the better result 

Stars path are loss function.
Global Minimum - least possible loss from our neural network

Red Circles then we move from it to the downwards global minimum and the process is called
gradient descent. it tells us what DIRECTION we need to move our function to determine or
get the global minimum.

Back propogation - goes backwards to the network updates the weights and bias according to the 
gradient descent and guide to the direction

NEURAL NETWORKS
INPUT - HIDDEN - OUTPUT LAYER
WEIGHTS ARE CONNECT TO EACH LAYER
BIAS (VALUE = 1) ARE ONLY CONNECTED TO THE HIDDEN AND OUTPUT LAYER, MOVE UP ,MOVE DOWN
THIS BIASES CAN BE THOUGHT OF Y-INTERCEPT THAT DEPENDS ON ACTIVATION FUNCTION


Activation Functions Roles
tanh = between -1 to 1
sigmoid = between 0 and 1
relu = 0 to positive infinity

WE ADD BIASES TO EACH WEIGHTED AND INPUT 
WE COMPARATIVELY LOOK THE OUTPUT 


data collection
defining the problem
inputs and weights and bias
calcuate for each neurons
apply the activation functions
loss functions 
backpropagation
descent gardient
training
validating
testing

BETTER THE PREDICTION LOWER THE LOSS FUNCTION
AS WE TRAIN OUR NETWORKS IT MAY GET BETTER OR WROST BASED ON
HOW WE CHOOSE WEIGHTS AND BIAS AND ACTIVATION FUNCTIONS,
OPTMIZATION AND GRADIENT DESCENT VALUES.

""
Binomial Distribution
   - it is not a continous data but a discrete data distribution eg coin toss
   - it has three parameters n, p and size
"""
# n = number of trails and p = probability of occurance and size = shape of the returned array
x = random.binomial(n=10, p= 0.5, size=10)
print("example of random binomial ", x)

# example of random binomial  [6 3 7 3 3 3 4 5 6 6]
"""
6: In the first trial, out of 10 attempts, there were 6 successes.
3: In the second trial, out of 10 attempts, there were 3 successes.
7: In the third trial, out of 10 attempts, there were 7 successes.
3: In the fourth trial, out of 10 attempts, there were 3 successes.
and so on ...
"""

# visualization of Binomial Distribution
# attribute kde=False: Specifies that a kernel density estimate (KDE) plot should not be overlaid on the histogram.
sns.distplot(random.binomial(n = 10, p= 0.5, size=1000), hist=True, kde=False, color="g")
plt.title("Binomial Distribution ")
plt.show()

# Visualization of Poisson Distribution
"""
Poisson Distribution is also a discrete distribution
but it handle or predict the number of time events will likely to occur in a specified time
it has two parameters 
lam = rate or known of number occurance
size = yes shape of the returned array
"""
y = random.poisson(lam=2, size = 10)
print("Poisson Dist --> ", y)

sns.distplot(random.poisson(lam=4, size=1000), kde=False, color="red")
plt.title("Poisson Distribution ")
plt.show()

# The first number, 6, represents the number of events in the first interval.
# The second number, 4, represents the number of events in the second interval.
# And so on, for each of the 15 intervals.

# Generate random samples
# loc -central values ,scale standard deviation 
normal_data = np.random.normal(loc=50, scale=7, size=1000)
poisson_data = np.random.poisson(lam=50, size=1000)
binomial_data = np.random.binomial(n=100, p=0.5, size=1000)

# Plot distributions
sns.distplot(normal_data, hist=False, label='Normal')
sns.distplot(poisson_data, hist=False, label='Poisson')
sns.distplot(binomial_data, hist=False, label='Binomial')

# Show legend
plt.legend()
# Show plot
plt.show()