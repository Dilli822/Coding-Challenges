
"""
RNN Continues - NLP 

Sequence of Data
In the previous tutorials we focused on the data that we could represent
as one static data point where the notion of time or step was irrelevant
Take for example our image data, it was simply a tensor of shape
(width, height, channels). That data doesnot change or care about the
notion of time.

In this tutorial we will look at sequence of text and learn how we can
encode them in a meaningful way. Unlike images, sequence data such as
long chains of text, weather patterns, videos and really anything where
the notion of a step or time is relevant needs to be processed and handled
in a special way.

But what do I mean by sequences and why is text data a sequence? Well
that's a good question. Since textual data contains many words that 
follow in a very specific and meaningful order we need to be able 
to keep track of each word and when it occurs in the data. Simply
encoding say an entire paragraph of text into one data point wouldn't
give us a very meaningful picture of the data and would be very difficult
to do anythin with. This is why we treat text as a sequence and process
one word at a time. We will keep track of where each of these words appear
and use that information to try to understand the meaning of pieces of text.

"""
# How our RNN going to read and actually understand and process the text/paragraph?
# Textual data into numeric data that can be feed into machine learning model.

# Tracking 
# Words - 0
# Hello - 1
# I - 2
# am - 3
# Dilli Hang Rai - 4
# from - 5
# Itahari - 6
# I - 1
# .
# .
# .
# .

# Dictionary Vocabulary of Words:
# Every unique word in our data set is the vocabulary and the
# model expects to see them as dictionary of words. 
# Every single one of these words, so every single one of these words in the vocabulary is going to be placed in a dictionary. And single one of these words in the vocabulary is going to be.
# placed in a dictionary. And beside that we are going to have some integer that represents it. So, for example, maybe the vocabulary of our data set is the words I, am, Dilli, Hang, Rai.
# We are going to keep track of the words that are present and the frequency of those words. And
# In fact, what we'll do well is we'll create what we call a bag and whenever we see a word appears,
# we'll simply add its number into the bag.
# There can be 1000 and millions of words, each has unique integers associated we focus on the track of the frequency and as bag of words becomes bigger, we will be losing the order and increasing
# frequency.


# Limitation of Bag of words:
# With the complex words, text there are words that have specific meaning, and this method is pretty flawed way to encode this data.
# Note: These are only rough explanation of Bag of words.

# Example on how bag of words lose the context or the real actual meaning of the
# sentences for an e.g.:
# I though the movie was going to be bad, but it was actually amazing!
# I thought the movie was going to be amazing, but it was actually bad!

# Although these two sentences are very similar, we know that they have different meanings. This is because of the ordering of words, a very important property of textual data.
# Now keep that in mind while we consider some different ways of encoding our textual data.

def bag_of_words(text):
    words = text.lower().split(" ") 
    bag = {}  # stores all of the encodings and their frequency 
    vocab = {}  # stores word to encoding mapping
    word_encoding = 1  # starting encoding
    
    for word in words:
        if word in vocab:
            encoding = vocab[word]
        else:
            vocab[word] = word_encoding
            encoding = word_encoding
            word_encoding += 1
        if encoding in bag:
            bag[encoding] += 1
        else:
            bag[encoding] = 1
    
    return bag, vocab

text = "Hello. I am Dilli Hang Rai. AI enthusiast from Nepal. I am doing my bachelors in Computer Science"
bag, vocab = bag_of_words(text)
print("Bag of words:", bag)
print("Vocabulary:", vocab)

# Output
# Bag of words: {1: 1, 2: 2, 3: 2, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 
# 14: 1, 15: 1, 16: 1}
# Vocabulary: {'hello.': 1, 'i': 2, 'am': 3, 'dilli': 4, 'hang': 5, 'rai.': 6, 'ai': 7, 
# 'enthusiast': 8, 'from': 9, 'nepal.': 10, 'doing': 11, 'my': 12, 'bachelors': 13, 'in': 14, 
# 'computer': 15, 'science': 16}

# Algorithm 
# Input: Get the text you want to analyze.

# Preprocessing: Convert text to lowercase and split it into words.

# Initialization: Set up an empty bag of words and an empty vocabulary.

# Loop through words:

# For each word:
# If the word is new, assign it a unique ID and add it to the vocabulary.
# Update the bag of words by counting the occurrences of each word ID.
# Output: Return the bag of words and the vocabulary.

# End.

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

# Zipf's Law: In a collection, the nth common term is 1/n times of the most common term. 
# E.g. the 5th most common word in English occurs nearly 1/5 times as often as the most common word.
#  Zipf distribution and its parameters can be valuable in various fields, particularly in linguistics,
#  statistics, and data science, where analyzing the frequency distribution of elements is important.

zip_dist = random.zipf(a = 2, size = (2, 3))
print("Zip Distribution Values : ", zip_dist)
# Zip Distribution Values :  [[1 1 3] [9 2 1]]

graph_zipf = random.zipf(a = 2, size = 900)
sns.distplot(graph_zipf[graph_zipf<10], kde=False, color='r')
plt.title("Zipf Distribution")
plt.xlabel("Element")
plt.ylabel("Frequency")
plt.show()