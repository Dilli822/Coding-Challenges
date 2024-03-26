

"""

Continue tutorial video 10:00 

Previously we just looked at the recurrent layer was called a simple RNN layer. Now we're going to talk about the 
layer which is LSTM which stands for "Long Short Term Memory". Now long and short are hyphenated together. But 
essentially what we're doing, and it just gets a little bit more complex, but I won't go into the Math is we add
another component that keeps track of the internal state. So right now, the only thing that we were tracking as 
kind of our internal state as the memory for this model was the previous output. So, whatever the previous output
was, so for example, at time zero here, there was no previous output, so there was nothing being kept in this model.

But at time, one, the outout from this cell right here was what we were storing.And then at cell two, the only thing
we were storing was the output at time one, right and we've lost now the output from time zero, what we're adding
in long short term memory is an ability to access the output from any previous state at any point in the future when
we want it. 

Now, what this means that rather than just keeping track of the previous output, we'll add all of outputs
that we've seen so far into what I'm going to call my little kind of " Converyor Belt", it's going to run at  the top
up here, I know it's kind of hard to see, but it's just what I'm highlighting, it's almost just like a lookup table
that can tell us the output at any previous cell that we want. So we can kind of add things to this Conveyor Belt,
we can pull things off, we can look at them. 

And this just adds a little bit of complexity to the model, it allows us to not just remember the last state, but
look anywhere at any point in time, which can be useful. Now, I don't want  to go into much more depth about exactly
how this works. 

But essentially, you know, just think about the idea that as the sequence gets very long, it's pretty easy to forget
the things we saw at the beginning. So if we can keep track of some of the things we've seen at the beginning, and 
sime of the things in between on this little conveyor belt, and we can access them whenever we want, then that's going
to make this probably a much more useful layer, right? 

We can look at the first sentence and the last sentence of a big piece of text at any point that we want, and say, okay,
you know, this text tells us about the meaning of this text, right. So that's what this LSTM does,.
I again, I don't want to go too far, we've already spent a lot of time kind of covering, you know, recurrent layers and 
how all this works. 

Anyways, if you do want to look it up some great mathematical definitions, again, I will source everything at the bottom 
of this document, so you can go there. But again, that's LSTM .
Although Simple RNN fairly work well for shorter length sequences. And again, remember we are treating our text as a 
sequence. Now, we're going to feed each word into the recurrent layer and it's going to slowly start to develop an 
understanding as it reads through each word right and processes that.

"""

# https://www.freecodecamp.org/learn/machine-learning-with-python/tensorflow/natural-language-processing-with-rnns-sentiment-analysis

"""
Natural Language Processing with RNNs: Sentiment Analysis:

Sentiment Analysis
And now time to see a recurrent network in action. For this example, we are going to do something called sentiment analysis.
The formal defintion of this terms from wikipedia is as follows:
The process of computationally identifying and categorizing expressed in a piece of text, especially in order to determine
whether the writer's attitude towards a particular topic, product, etc is positive, negative, or neutral.

The example well use here is classifying moview reviews as either positive, negative or neutral.

This guide is based on the following tensorflow tutorial:
https://www.tensorflow.org/tutorials/text/text_classification_rnn


Moview Review Dataset
Well start by loading int he IMDB movie review dataset from keras. This dataset contains, 25,000 reviews from the IMDB 
where each one is already preprocessed and has a label as either positive or negative. Each review is encoded by integers 
that represents how common a word is in the entire dataset. For example a word encoded by the integer 3 means that it is 
the 3rd most common word in the dataset.
=========================================================================================================
From tutorial:
It contains 25,000 reviews, which are already pre-processed and labeled. Now what means for us is that every single word 
is actually already encoded by an integer. And in fact, they've done kind of a clever encoding system where what they've 
done is said, if a character is encoded by say, integere 0, that represents how common that word is in the entire data set.
So if an integer was encodede by or non-integer.
# Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz

"""

from keras.datasets import imdb
from keras.preprocessing import sequence
import tensorflow as tf
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


VOCAB_SIZE = 88584
MAXLEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)
train_data[0]
len(train_data[0]) # every review either should have unique length
print("train data --> ", train_data[0])
print("train data length ---> ", len(train_data[1]))
print("train data type --> ", type(train_data[0]))

# Calculate lengths of the first five training samples
train_lengths = [len(sample) for sample in train_data[:5]]

# Calculate the number of unique values in the first five training samples
unique_values = [len(set(sample)) for sample in train_data[:5]]

# Create a scatter plot for values of the first five samples
plt.figure(figsize=(8, 6))
for i, sample in enumerate(train_data[:5], 1):
    plt.scatter([i] * len(sample), sample, color='green', alpha=0.85)  # Scatter plot for values of each sample
plt.title('Values of First Five Training Samples')
plt.xlabel('Sample Number')
plt.ylabel('Values')
plt.xticks([1, 2, 3, 4, 5], [f'Sample 1\n({len(train_data[0])} values)', 
                             f'Sample 2\n({len(train_data[1])} values)',
                             f'Sample 3\n({len(train_data[2])} values)', 
                             f'Sample 4\n({len(train_data[3])} values)', 
                             f'Sample 5\n({len(train_data[4])} values)'])
plt.show()

# Create a scatter plot for lengths of the first five samples
plt.figure(figsize=(8, 6))
plt.scatter(range(1, 6), train_lengths, color='red')  # Scatter plot for lengths
plt.title('Lengths of First Five Training Samples')
plt.xlabel('Sample Number')
plt.ylabel('Length')
plt.xticks(range(1, 6), [f'Sample 1\n({train_lengths[0]} length)', 
                         f'Sample 2\n({train_lengths[1]} length)',
                         f'Sample 3\n({train_lengths[2]} length)', 
                         f'Sample 4\n({train_lengths[3]} length)', 
                         f'Sample 5\n({train_lengths[4]} length)'])
plt.show()

