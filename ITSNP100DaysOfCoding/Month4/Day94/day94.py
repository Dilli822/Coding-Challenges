
"""
You may somethimes see the term optimizer or optmization function. This is simply the function that implements the backprogagation algorithm descrived above.
Here's a list of a few  common ones.
1. Gradient Descent
2. Stochastic Gradient Descent
3. Mini-Batch Gradient Descent
4. Momentum
5. Nesterov Accelerated Gradient

* This article explain them quite well is where I've pulled this list from +
https://medium.com/@sdoshi579/optimizers-for-training-neural-network-59450d71caf6

"""

""""
Creating a neural network
Okay now you have reached the exciting part of this tutorial! No more math and complex algorithm. Time to get hands on and train a very basic neural network. As
stated earlier this guide is based off of the following TensorFlow tutorial:
https://www.tensorflow.org/tutorials/keras/classification

"""

# Imports
import tensorflow as tf
from tensorflow import keras 

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Dataset - for this tutorial we will use the MNIST Fashion Dataset. This is a dataset that is included in keras.This dataset includes 60,000 images for training
# and 10,000 images for validation/testing. 

fashion_mnist = keras.datasets.fashion_mnist # load dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # split into testing and training 

# let's have a look at this data to see what we are working with 
train_images.shape

# So we've got 60,000 images that are made up of 28x28 pixels(784 in total)
train_images[0, 23, 23] # let's have a look at one pixel 

# OUR pixel values are  between 0 and 255, 0 being black and 255 being white. This means we have a grayscale image as there are no color channels
train_labels[:10] # let's have a look at the first 10 training labels

# Our labels are integers raning from 0 -9 .Each integer representing a specific article of clothing. We'll create an array of labels names to indicating which is 

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Finally let's look at what some of these images look like
plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()

"""

Data Preprocessing:
The last step before creating our model is to preprocess our data. This simply means applying some prior transformations to our data before feeding it the model. In this 
case we will simply scale all of our greyscale pixel value(0-255) to be between 0 and 1. We can do this by dividing each value in the training and testing sets by 255.0.
We do this because smaller values will make it easier for the model to process our values.


"""

train_images = train_images / 255.0
test_images = test_images / 255.0 

print(test_images)


# Building the Model:
# Now it's time to build the model! we are going to use a keras sequential model with 3 different layers. This model represents a feed-forward neural network
# (one that passes values from left tp right), we'll break down each layer and it's architecture below.

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (100, 100)), # input layer 1
    keras.layers.Dense(128, activation = 'relu'), # hidden layer 2
    keras.layers.Dense(10, activation ='softmax') # output layer 3
])

print(model.summary)

"""
Layer 1: This is our input layer and it will consist of 784 neurons. We use the flatten layer with an input shape of (28, 28) to denote that our input should 
come in that shape. The flatten means that our layer will reshape the shape(28, 28) array into a vector of 784 neurons so that each pixel will be associated 
with one neuron.

Layer 2: This is our first and only hidden layer. The dense denotes that this layer will be fully connected and each neuron from the previous neural connects
to each neuron of this layer. It has 128 neurons and uses the rectify linear unit activation function.

Layer3: This is our output layer and also a dense layer. It has 10 neurons that will look at to detemine our models output. Each neuron represents the
probability of a given image being one of the 10 different classes. The actication function softmax is used on this layer to calculate a probability 
distribution for each class. This means the value of any neuron in this layer will be between 0 and 1 where 1 represents high probability of image being in 
that class.

"""

# Assuming train_images is a list or an array containing images
num_images_to_show = 10
last_images = train_images[-num_images_to_show:]

# Create a figure and axes
fig, axes = plt.subplots(2, 5, figsize=(10, 5))  # 2 rows, 5 columns

# Flatten the axes array
axes = axes.flatten()

# Loop through images and plot them
for i in range(num_images_to_show):
    axes[i].imshow(last_images[i], cmap=plt.cm.binary)
    axes[i].set_title(f"Image {i+1}")
    axes[i].axis('off')  # Turn off axis lines and labels

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()



