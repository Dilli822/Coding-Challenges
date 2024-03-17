"""
-- Pretrained Model 
-- We would have noticed that the model takes a few minutes to train in the
   Notebook and only giv'es an accuraccy of ~ 70% this is okay but surely
   there is a way to improve on this.
   In this section, we will talk about using a pretrained CNN as apart of
   our own custom network to improve the accuracy of our model. We know
   that CNN's alone (with no dense layers) donot do anything that map the
   presence of features from our input. This means we can use a pretrained 
   CNN, one trained on millions of images, as the start of our model. This
   allows us to have a very good classifier for a relatively smaller
   dataset ( < 10, 000 images). This is because the convnet already has
   a very good idea of what features to look for an image and can find 
   them very effectively. So if we can determine the presence of features
   that all the rest of the model needds to do is determine which combination
   of features makes a specific image.
   
   
   Explaination:
   We have used augmenation, that's great technique if we want to increase
   the size of our data set. But what if even after that we still donot
   have enough images in our dataset?
   well, what we have can do is use something called a pre trained model
   now companies like google, and tensorflow, which is owned by Google
   make their own amazing compositional neural networks that are completely
   open source that we can use. So what we are going to do is actually use
   part of a Convolutional neural network that they've trained already on,
   I guess 1.4 million images,And we are just going to use part of that 
   model, as kind of the base of our models so that we have a really good
   starting point. And all we need to do is called fine tune the last few
   layers of that networkm so that they work a little bit better for our
   purposes. So what we are going to do essentially say, all right 
   we have this model that Google's trained they've trained it on 1.4
   million images its capable of classifying, let's say 1000 different classes
   which is actually the example we'll look at later.
    
"""


"""
   Fine Tuning 
   
   When we employ the techique defined above we will often want to tweak the 
   final layers in our convolutional base to work better for our specific
   problem. This involves not touching or retraining the earlier layers in 
   our convolutional base but only adjusting the final few. We do this
   because the first layers in our base are very good at extracting low
   level features like lines and egdes, things that are similar for any 
   kind of image. 
   where the later layers are better at picking up very specific features
   like lines and edges, things that are similar for any kind of image.
   where the later layers are better at picking up very specific features
   like shapes or even eyes. 
   >>>> If we adjust the final layers that we can look for only features 
   relevant to our specific problem.
   
   
   Explanation Continues....... 
   
   So obviously the begining of that model, is what's picking up on the
   smaller edges, and you know kind of the very general things that appear
   in all of our images. So if we can use the base of that model, so kind 
   of the  begining of it, that does a really good job picking up on edges,
   and general things that will apply to any image, then what we can do
   is just change the top layers of that model a tiny bit, or add our own
   layers to it to classify for the problem that we want. And that should be
   a very effective wy to use this pre trained model, we are saying we are
   going to use the begining part that's really good at kind of the 
   generalization step, then we'll pass it into our own layets that will
   do whatever we need to do specifically for our problem. That's what's 
   like fine tuning step. And then we should have a model that works pretty 
   well. in fact that's what we are going to do in this example, now.
   
   So that's kind of the point of what I am talking about here is using 
   part of a model that already exists that's very good at generalizing
   it's been trained on so many different images. And then we'll pass 
   our own training data in, we won't modify the beginning aspect of 
   our neural network, because it already works really well, we'll
   just modify that last few layers that are really good at classifying,
   for example, just cats and dogs, which is exactly the example
   we are actually going to do here. 

"""

# Using a pretrained model
# Link tutorial https://www.tensorflow.org/tutorials/images/transfer_learning
import os
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras 

# Datasets - load cats vs dogs dataset from module
# tensorflow dataset contains image and label pairs where
# images have different dimensions and 3 color channels
# import tensorflow_dataset as tfds
# tfds.disable_progress_bar()

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

# split the data manually into 80% training, 10% testing
# and 10% validation
# (raw_train, raw_validation, raw_test) , metadata = tfds.load(
#     'cats_vs_dogs',
#     split=[
#         'train[:80%]', 
#         'train[80%:90%]',
#         'train[90%]'
#     ],
#     with_info = True,
#     as_supervised = True,
# )
# except 4 all will occur size = 100 times runs this dist
data_dist = random.choice([1,2,3,4], 
                          p = [0.2, 0.5, 0.3, 0.00],
                          size = 100)

# gives normal distribution of data with loc,scale and size param
# scale - stdeviation, loc = mean/peak
normal_dist = random.normal(loc=1,scale = 2,size=(2,3))

# discrete distribution, outcomes toss of a coin
binomial_dist = random.binomial(n = 10,p = 0.5,size=10)

# n times an event can happen in a specified time
# lam rate of occurance
poisson_dist = random.poisson(lam=2, size=1000)

# equal chance of occuring
uniform_dist = random.uniform(size=1000)

# describe the growth rate
logistic_dist = random.logistic(loc=1, scale=2, size=(2, 3))

# generalizaation of binomial distribution
multinomial_dist = random.multinomial(n=6, pvals=[1/6, 1/6, 1/6, 1/6,1/6])

# basis to verify the hypothesis
chisquare_dist = random.chisquare(df=2, size=(2,3))
# time till next event success or failures
exponential_dist = random.exponential(scale=2, size=(2,3))

import seaborn as sns
import matplotlib.pyplot as plt
# Create subplots with Seaborn
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

# Plot each distribution
sns.kdeplot(data_dist, ax=axes[0, 0], color='blue')
sns.kdeplot(normal_dist.flatten(), ax=axes[0, 1], color='orange')
sns.kdeplot(binomial_dist, ax=axes[0, 2], color='green')
sns.kdeplot(poisson_dist, ax=axes[1, 0], color='red')
sns.kdeplot(uniform_dist, ax=axes[1, 1], color='purple')
sns.kdeplot(logistic_dist.flatten(), ax=axes[1, 2], color='brown')
sns.kdeplot(multinomial_dist.flatten(), ax=axes[2, 0], color='pink')
sns.kdeplot(chisquare_dist.flatten(), ax=axes[2, 1], color='gray')
sns.kdeplot(exponential_dist.flatten(), ax=axes[2, 2], color='cyan')

# Set titles
axes[0, 0].set_title('Data Distribution')
axes[0, 1].set_title('Normal Distribution')
axes[0, 2].set_title('Binomial Distribution')
axes[1, 0].set_title('Poisson Distribution')
axes[1, 1].set_title('Uniform Distribution')
axes[1, 2].set_title('Logistic Distribution')
axes[2, 0].set_title('Multinomial Distribution')
axes[2, 1].set_title('Chi-square Distribution')
axes[2, 2].set_title('Exponential Distribution')

# Adjust layout
plt.tight_layout()
# Show the plot
plt.show()
