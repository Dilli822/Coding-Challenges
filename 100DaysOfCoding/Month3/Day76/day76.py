
# Object Detection and Recognition
# Tensorflow Object Detection API 
# https://github.com/tensorflow/models/tree/master/research/object_detection

# ==========================================================================
# Natural Language Processiong (NLP)
# text,textual data, paragraphs, sentences,words are classified under NLP when merged with Machine Learning Stuffs, 
# language processed by Machine for  human readable and understandable format instead for machine itself. 
# Natural Language Processing or(NLP for short) is a discipline in computing
# that deals with the communication between natural (human) languages and computer computer languages. 
# A common example of NLP is something like spellcheck or autocomplete. Essentially NLP is the field that 
# focuses on how computers can understand and/or process natural/human languages.

# ==========================================================================
# Recurrent Neural Networks
# In this tutorial we will introduce a new kind of neural network that is much more capable of processing 
# sequential data such as text or characters called a recurrent neural network (RNN for short).
# We will learn how to use a reccurrent neural network to do the following:
# 1. Sentiment Analysis
# 2. Character Generation

# RNN's are fairly complex and come in many different forms so in this tutorial. 
# we will focus on how they work and the kind of problems they are best suited for.

# >>> we are skipping the fundamental maths part and focus why this works the way it does
# rather than how and we will know when we should use this.
# and of course some layers of RNN.for more details or math we have link 

"""
What we will be doing?
Sentiment Analysis -> use movie reviews and try to determine whether these moview reviews 
                      are positive or negative by performing analysis on them.
                      
Note: Sentiment Analysis tells us the text,words,sentence is positive or negative.

Character Generation -> to generate the next character in a sequence of 
                           text for us. And we are going to use that model a 
                           buch of times to actually generate an entire play.
                
Note: Train the model to lean how to write a play.

Task: Model will read the play Romeo and Juilet and able to write the play
text. We give it a little promplt when we're actually using the model and 
say okay this is the first part of the play, write the rest of it, and 
then it will actually go and write the rest of the characters in the play.

And we'll see that we can get something that's pretty good using the 
techniques that we'll talk about.

Data types -> Numeric and Textual data how we are able to make neural network 
to understand the whole text and paragraph how data types are processed.
How can we turn some textual data into numeric data that we can 
feed to our neural network. Let's understand using drawing charts.

"""
# ================================================================
# Bag of Words 
# Encode and Pre-Process Text into Integers
# Method to convert textual data to numeric data
# This algo only works for simple task or text pretty flawed

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import seaborn as sns

# Rayleigh Distribution 
# - used in signal processing
# - has two parameters: scale(standard deviation) default is 0 how flat it is 
#                       size - returned array

x = random.rayleigh(scale=2, size=(2, 3))
print("rayleigh is used in signal processing -->", x)

# Visualization
# At unit stddev and 2 degrees of freedom rayleigh and chi square represent the same distributions.
sns.distplot(random.rayleigh(size=1000), hist=False, color="g")
plt.title("Rayleight Distribution")
plt.show()

# Pareto Distribution
# A distribution following Pareto's law i.e. 80-20 distribution (20% factors cause 80% outcome).
# a - shape parameter

y = random.pareto(a=2, size=(2, 3))
print("Pareto distribution ", y)

sns.distplot(random.pareto(a=2, size=1000), kde=False)
plt.title("Pareto Distribution")
plt.show()