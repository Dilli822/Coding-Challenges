
# Using a pretrained model
# Link tutorial https://www.tensorflow.org/tutorials/images/transfer_learning
# https://chat.openai.com/share/387640c6-f467-4f5c-acce-b3d43ad6260c
import os
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras 
from tensorflow.keras.models import Sequential


# Datasets - load cats vs dogs dataset from module
# tensorflow dataset contains image and label pairs where
# images have different dimensions and 3 color channels
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
# split the data manually into 80% training, 10% testing
# and 10% validation
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

get_label_name = metadata.features['label'].int2str
# creates a function object that we can use to get labels

# display 2 images from the dataset 
for image, label in raw_train.take(1):
    plt.figure()
    plt.imshow(image)
    plt.title("Original Dataset " + get_label_name(label))

plt.show()

# lets reshape the images
# smaller the image better it is in compressed form
IMG_SIZE = 160 # All Images will be resized to 160 x 160

def format_example(image, label):
    """
    returns an image that is reshape to IMG_SIZE
    """
    image = tf.cast(image, tf.float32)
    image = (image/ 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label 

# Now apply this function to all our image using map
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

# lets have a look at our images now
for image, label in train.take(1):
    plt.figure()
    plt.imshow(image)
    plt.title("Training Dataset " + get_label_name(label))
    
plt.show()

# Now let's have a look at the shape of an original image vs the 
# new image we will see if it has been changed

for img, label in raw_train.take(2):
    print("Original Shape: ", img.shape)

for img, label in train.take(2):
    print("New shape: ", img.shape)

# output
# Original Shape:  (262, 350, 3)
# Original Shape:  (409, 336, 3)
# New shape:  (100, 100, 3)
# New shape:  (100, 100, 3)


"""
Picking a PreTrained Model
The model we are going to use as the convolutional base for
our model is the >> MobileNet V2 developed at Google. 
This model is trained on 1.4 million images and has
1000 different classes.

We want to use this model but only its convolutional base.
So wehn we load in the model we'll specify that we donot
want to load the top (classification) layer. We'll tell 
the model what input shape to extract and to use the 
predetermined weights from imagenet(Google dataset)

"""
# here 3 is color channel
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(
    input_shape = IMG_SHAPE, # 160 x 160
    include_top = False, # do we include classifier ?
    weights = 'imagenet' # imagenet specific shape of weight
)

base_model.summary()

# We are using this because we could not make it alone
# Ouput All this NN are the result of working by Experts,Ph.d SCholar

"""

output is : 
out_relu (ReLU)  (None, 5, 5, 1280)  0  ['Conv_1_bn[0][0]']
This o/p is actual useful and we gonna take this and pass
that to some more convolutional layers, and actually our 
classifier and use that to predict dogs versus cats.

At this point this base_mode will simply output a shape
(32. 5, 5, 1280) tensor that is a feature extraction
from our original (1, 160, 160, 3) image. This 32 means 
that we have 32 layers of different filters/features.

"""


# for image, _i in train_batches.take(1):
#     pass

# feature_batch = base_model(image)
# print(feature_batch.shape)
# # expected output is (32, 5,5,1280)


"""
Frezing base model
==================================================================================================
Total params: 2257984 (8.61 MB)
Trainable params: 2223872 (8.48 MB)
Non-trainable params: 34112 (133.25 KB)

- So essentially we wanna use the MobileNet V2 as the base
of our network, which means we don't want to change it. 
If we just put this network in right now is the base to
our neural network. Well, what's going to happen is, it's 
going to start retraining all these weights and biases and 
in fact, it's going to train 2.257 million more weights
and biases, when in fact, we don't want to change these
because these have already been defined, they been set.
And we know that they will work well for the problem
already.
They worked well for classifying 1000 classes. Why are 
we going to touch this now.
We donot want to train this, we want to leave it the same.
So to do that, we are just gonna freeze it.

Freezing is a pretty, I mean it just essentially means
turning the trainable attribute of a layer off or of the
model off.

"""

# setting base model False
base_model.trainable = False
base_model.summary()

"""
Before making base model True
==================================================================================================
Total params: 2257984 (8.61 MB)
Trainable params: 2223872 (8.48 MB)
Non-trainable params: 34112 (133.25 KB)

After making the base model False
==================================================================================================
Total params: 2257984 (8.61 MB)
Trainable params: 0 (0.00 Byte)
Non-trainable params: 2257984 (8.61 MB)
__________________________________________________________________________________________________

"""

# Adding Own Classifiers
# Lets add our own classifiers on top of this.
"""
last output:
out_relu (ReLU) (None, 5, 5, 1280)  0 ['Conv_1_bn[0][0]']           
         
>> take this above output 5x5 and we want to use it 
   to classify either Cat or either Doc, right so what we
   are going to do is add a global average layer, which
   essentially is going to take the entire 
                
Now that we have base layer setup we can add the classifier
instead of flatterning the feature map of the base layer
average of every single so of 1280 different layers that
are 5 x 5 PUT that into a one D Tensor, which is kind 
of flattening that for us. So we do that global average
pooling. And then we are just going to add the prediction
layer, which essentially is going to just be one dense
node. And since we are only classifying two different
classes, right dogs and cats, we only need one then we
are goinf to add all these model together. So the base
model is layers, the global average layer that we define
there, and then the prediction layer to create our final
model.

model = keras.layers.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])


we will use a global average pooling layer that will 
average the entire 5x5 area of each 2D feature map and 
return to us a single 1280 element vector per filter.


"""

# global average -  extracting meaningful features from the pre-trained convolutional base and preparing them for the final classification layers.
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

# Finally we will add the prediction layer that will be 
# a single dense neuron. We can do this because we only
# have two classes to predict for

# prediction layer 
prediction_layer = keras.layers.Dense(1)

# model layer
# -------- Magic Happens Here --------
from tensorflow.keras.models import Sequential

# Create the Sequential model
model = Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

model.summary()


"""
__________________________
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 mobilenetv2_1.00_160 (Func  (None, 5, 5, 1280)        2257984   
 tional)                                                         
                                                                 
 global_average_pooling2d (  (None, 1280)              0         
 GlobalAveragePooling2D)                                         
                                                                 
 dense (Dense)               (None, 1)                 1281      
                                                                 
=================================================================
Total params: 2259265 (8.62 MB)
Trainable params: 1281 (5.00 KB)
Non-trainable params: 2257984 (8.61 MB)
____________________________________________________

Breakdown of output 
 mobilenetv2_1.00_160 (Func  (None, 5, 5, 1280)        2257984   
 tional)    
 This is our base layer and that's fine because the op shape
 is that then 
 global_average_pooling2d ( GlobalAveragePooling2D) (None, 1280)  0                                
 global average pooling which again 
 just flattens it out does the average for us. And then

 dense (Dense)               (None, 1)                 1281    
 finally our dense layer which is going to simply have one
 neuron which is going to be our output.
 
 
 Total params: 2259265 (8.62 MB)
 Trainable params: 1281 (5.00 KB)
 Non-trainable params: 2257984 (8.61 MB)
 Here only 1281 is trainable params because connections
 from 
  global_average_pooling2d ( GlobalAveragePooling2D) (None, 1280)  0                                
  to this layer
   dense (Dense)  (None, 1) 1281    
   
   that means 1280 weights and one bias.So that is what we
   are doing and what we have created. Now this base, the
   majority of the network has been for us. And just add
   our little classifier on top of this. 
   we 
   

  
"""
# Plot the original and resized image

import tensorflow as tf
from tensorflow.keras import layers

# Define the model architecture
model = tf.keras.Sequential([
    layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE, 3)),  # Flatten the input images
    layers.Dense(128, activation='relu'),  # Dense layer with 128 neurons and ReLU activation
    layers.Dropout(0.5),  # Dropout layer to prevent overfitting
    layers.Dense(1, activation='sigmoid')  # Output layer with 1 neuron for binary classification
])

# Compile the model
model.compile(optimizer='adam',  # Adam optimizer
              loss='binary_crossentropy',  # Binary cross-entropy loss function
              metrics=['accuracy'])  # Track accuracy during training

# Display the model summary
model.summary()


import matplotlib.pyplot as plt

# Define the neural network architecture
architecture = [
    {"layer": "Input", "neurons": 160*160*3},
    {"layer": "Global Average Pooling", "neurons": 1280},
    {"layer": "Dense", "neurons": 1}
]

# Plot the architecture
plt.figure(figsize=(10, 5))
for i, layer in enumerate(architecture):
    plt.barh(i, layer["neurons"], color='blue', alpha=0.5)
    plt.text(layer["neurons"] + 10, i, f'{layer["layer"]}\n{layer["neurons"]} neurons', va='center')
plt.xlabel('Neurons')
plt.title('Neural Network Architecture')
plt.yticks(range(len(architecture)), [layer["layer"] for layer in architecture])
plt.gca().invert_yaxis()  # Invert y-axis to show input layer at the top
plt.tight_layout()
plt.show()

from tensorflow.keras.utils import plot_model
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Generate a visual representation of the model architecture
plot_model(model, to_file='model_visualization.png', show_shapes=True, show_layer_names=True)

# Display the visual representation directly
model_img = plt.imread('model_visualization.png')
plt.imshow(model_img)
plt.axis('off')  # Hide axis
plt.show()