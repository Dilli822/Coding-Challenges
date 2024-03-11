

"""
Optimizer
 - Optimizer or optimization function is simply function 
 that implements the back propagation algorithm.
 
 - LISTS
 1. Gradient Descent
 Link:
 https://medium.com/@sdoshi579/optimizers-for-training-neural-network-59450d71caf6

"""

# Deep Computer Vision

"""
Image Classification and Object detection/recognition using deep computer vision
with something we call a convolutional neural network.

The goal of our convoloutional neural networks will be to classify and detect images or specificobjects from within the image. 
We will be using image data as our features and a label for those images as our label or output. 

We already know how neural networks work so we can skip through the basics and move right into 
explaining the following concepts.
1. Image Data
2. Convolutional Layer
3. Pooling Layer
4. CNN Architectures

The major differences we are about to see in these types of neural networks are the layers that make them up.


1. Image Data 
   so far we have dealt with pretty straight forward data that has 1 or 2 dimensinos. Now we are about to deal with 
   image datais usually made up of 3 dimensions . These dimensions are as follows:
   
   - image height
   - image width
   - color channels
   The only item in the list above you may not understand is color channels. The number of color channels represents the depth of 
   an image and coorelates to the colors used in it. For example, an image with three channels is likely made up of rgb(red, green, blue)
   pixels. So Far, each pixel we have three numeric values  in the range of 0-255 that defines its color.For an image of color depth 1 we would
   likely have a greyscale image with one value defining each pixel, again in the range of 0-233.
   

 Convolutional Neural Network
 Note: I will use the term convnet and convolutional neural network interchangably.
 Each convolutional neural network is made up of one or many convolutional layers. These layers are different than the dense layers we have
 seen before. Their goal is to find the patterns from within images that can be used to classify the image or parts of it. But this may sound
 familiar to what our densly connected neural network in the previous section was doing, well that's because it is.
 
 The fundamental difference between a dense layer and convolutional layer is that dense layers detect patterns globally while convolutional 
 layers detects patterns locally. When we have a densely connected layer each node in that layyers sees all the data from the previous layer.
 This means that this layer is looking at ALL of the information and is only capable of analyzing the data in a global capacity. Our convolutional
 layer however will not be densly connected, this means it can detect local patterns using part of the input data to that layer.
 
 Let's have a look at how a densly connected layer would look at an image vs how a convolutional layer would.
 
 For an example in our cat image cat has left side head and image now our classification model
 memorized that part and Ah thats cat it has certain global pattern that model is memorizing it .
 But it would not recoginze if the same cat image is flipped horizontal our model gets confused.
 It learnt the pattern in the specific red dotted rectangle part only.
 
 But Convolutional network scan each line and find the features in the image and based on that features 
 it pass the features to the dense layers, now dense classifier based on that features,determing the combination of 
 these presences of features that make up specific classes or make up specific objects.
 
 SO the dense layers sees the pattern in the whole dense layer whereas convolutional will look in local areas notice the 
 features in local areas not just global.
 
"""
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

# Uniform Distribution - probability of having equal chances of occuring
# it has three parameters 
# a - lower bound - default 0.0
# b - upper bound - default 1.0
# size - the shape of the returned array
x = random.uniform(size=(2, 3))
print(x)

lower_bound = 5
upper_bound = 10
size = (4,2)
y = np.random.uniform(lower_bound, upper_bound, size)
print("uniform distribution is --> ", y)

""" 
uniform distribution is -->  [[5.03217856 6.87083107]
 [7.86341985 7.09328516]
 [8.89725236 8.91325541]
 [9.94681885 6.38931071]]
 
Since the upper bound is exclusive, the generated numbers can be any value from
5 (inclusive) to just below 10 (exclusive). Each time you run the code, you'll get
a different set of random numbers due to the nature of random number generation
"""

sns.distplot(random.uniform(size=4000), hist=False)
plt.title("Uniform Distribution shows equal chances of probability ")
plt.show()

# Logistic distribution - show the growth, used in ML in logistic regression, neural networks etc.
# Has three parameters 
# loc - mean the peak value and default value is 0
# scale - standard deviation, the flatness of distribution and the default is 1
# size - the shape of the returend array 

logistics_var = random.logistic(loc= 1, scale= 2, size=(2,3))
print("Logistics distribution  is --> ", logistics_var)

# Logistics distribution  is -->  [[ 2.24674803  1.8188065   0.16330728]
#  [-6.23776222 -0.22249202  1.8240681 ]] 
sns.distplot(random.logistic(size=2000), hist=False, color="r")
plt.title("LLogistics distribution  with mean 1 and scale is 2")
plt.show()

# Let's display multiple plots or sub plots
# first and second are row and col and last one is index
#plot 1:
x = np.array([0, 2, 4, 6])
y = np.array([1, 3, 5, 7])

# 1 row 2 columns 
plt.subplot(1, 2, 1)
plt.plot(x,y, color="red")
plt.title("indexed 0 subplot")

#plot 2:
x = np.array([ 1, 1, 3, 5])
y = np.array([2, 4, 6, 8])

# 1 row and 2 columns and keep index 1 to overlap
plt.subplot(1, 2, 2)
plt.plot(x,y,color="blue",marker='o')
plt.title("indexed 1 subplot")
plt.show()

