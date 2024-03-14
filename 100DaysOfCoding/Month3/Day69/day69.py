

"""
https://miro.com/app/board/uXjVNhgCCxA=/?share_link_id=748462891002

 ----- Pooling Operation --------
we have tons of layers and lots of computation for all this filter
and there must be someway to make these a little bit simpler
a little bit easier to use. Well, yes, that's true. And there is a 
way to do that And that's called pooling. So there's three
types of pooling

Max -mostly used that tell us about the maximum presence of a 
feature that kind of local area, we really only care if the 
feature exists.

Min - if 0 does not exists.

Average - not mostly used, tells average presence of features 
in local area.


A Pooling operation is just taking specific values from a sample of the
output feature map. So once we generate this output feature map,
what we do to reduce its dimensionality and just make it a little bit
easier to work with, is when we sample typically 2 x 2 areas of this
output feature map, and just take either the min max average value of 
all the values inside of here and map these, we are goona go back this
way to a new feature map that's twice the one times the size essentially.

What are the three main properties of each convolutional layer?
Input size, the number of filters, and the sample size of the filters.
"""


"""
Coding Tutorials Link:
https://www.tensorflow.org/tutorials/images/cnn

1. Create our first Convnet to get familiar with CNN architectures.
2. Dataset - CIFAR Image Dataset wll be used in tensorflow to classify 10
   different everyday objects. total images 60K with 32 x 32 images
   with 6K images of each class.
   
   Labels in the dataset are:
   ['airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck']
    
    Dataset:
    Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
   
   
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


# LOAD AND SPLIT THE DATASET
# loading tensorflow strange set of data objects
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize the pizel to between 0 and 1
# divide by 255 to make values between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names =    ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']

# We can look at a image
IMG_INDEX = 7 # CHANGE NUMBERS TO SEE THE IMAGE

plt.imshow(train_images[IMG_INDEX], cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels [IMG_INDEX] [0]])
plt.show()


# CHI - SQUARE DISTRIBUTION - used as a basis to verify
# the hypothesis. It has two parameters 
# df - degree of freedom
# size - the shape of the returned array

import numpy as np
from numpy import random
import seaborn as sns

x = random.chisquare(df=2, size=(2,3))
print("Chi square dist used as a basis to verfiy hypothesis --> ", x)

data = random.chisquare(df=1, size=1500)
sns.distplot(data, hist=False, color='g')
plt.title("Chi Square A chi-square test is a statistical test used to compare observed results with expected results")
plt.show()