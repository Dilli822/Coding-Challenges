


# ---------- HIDDEN MARKOV MODELS --------------------
# It is a finite set of states, each of which is associated with a (generally mutli-dimensional) probability distribution [].
# Transitions among the states are governed by a set of probabilities called transition probabilities. 
# 
# A hidden markov model works with probabilities to predict future events or states, we will learn how to create 
# a hidden markov model that can predict the weather

# --------- Data ---------

# bunch of states - cold day, hot day 
# type od data to work with a hidden markov model d
# we are only interested in probability distributions unlike 
# other model we are using 100% of dataset entries


# ---------------- Components of Markov Model ------------
# 1. States: a finite number of or a set of states, like "warm", "cold", "hot", "low", "red", 
# and SO ON THESE STATES ARE HIDDEN 

# 2. Observation: Each state has a particular observation associated with it based on a probability distribution. 
# An example if it is hot day then Dilli has 80% chance of being happy and 20% chance of being sad it is observation.

# 3. Transitions: Each state will have a probability defining the likelyhood of transitioning to a different state.
# an example is th following: a cold day has a 50% change of being followed by a hot day and a 50% chance of being followed by another cold day


import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# Shuffling Arrays

random.shuffle(arr)

print("shuffling an array ", arr)


arr2 = np.array([11, 22, 33, 44, 55])

print("random permtuation ", random.permutation(arr2))


x = random.randint(100)
y = random.rand()
print("random float numbers  ", y)
print("random integer numbers between 0 to 100  ",x)


normal_distribution = random.normal(size=(2, 3))

print("normal distribution points ", normal_distribution)

# Plotting histogram
plt.hist(normal_distribution.flatten(), bins=10, density=False, alpha=0.7, color='green', edgecolor='black')
plt.title('Histogram of Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(False)
plt.show()


# Bins: bins refers to the number of intervals or bins into which the data is divided in the histogram. 
# It determines the granularity of the histogram. For example, if bins=10, the data range will be divided 
# into 10 equally spaced intervals, and the histogram will display the frequency of values falling within each interval.

# Alpha: alpha controls the transparency of the histogram bars. It accepts values between 0 and 1, where 0 means 
# fully transparent (invisible) and 1 means fully opaque (solid). Setting alpha to a value less than 1 allows you to 
# see through the histogram bars, which can be useful for visualizing overlapping data or creating layered plots.

# Density: density is a boolean parameter that determines whether the histogram should be normalized to form a 
# probability density histogram. When density=True, the height of each bin is normalized such that the total area
# under the histogram equals 1, representing the probability density function. This can be useful for comparing 
# histograms of datasets with different numbers of samples or different ranges, as it accounts for differences in data density.

