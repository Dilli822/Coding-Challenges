

"""
Natural Language Processing with RNNs: Recurring Neural Networks

Now that we've learned a little but about how we can encode text it's time to dive into recurrent
neural networks. Up until this point we have been using something called "feed-forward" neural
networks. 
This simply means that all of our data is fed forwards(all at once) from left to right through the network.
This was fine for the problems we considered before but won't work very well for processing text.
After all even we (humans) don't process text al at once. We read word by word from left to right
and keep track of the current meaning of the sentence so we can understand the meaning of the next
word.

Well this is exactly what a recurrent neural network is designed to do. When we say recurrent neural
network all we really mean is "a network that contains a loop". A RNN will process one word at a tome
while maintaining an internal memory of what it's already seen. This will allow it to treat words differently
based on their order in a sentence and to slowly build an understanding of the entire input, one word
at a time.

This is why we are treating our text data as a sequence! So that we can pass one word at a time to the
RNN.
Let's have a look at what a recurrent layer might be like:

"""


"""
From tutorial:
Previously, we have talked about kind of the form that we need to get our data in before we can pass it
further in the Neural Network right before it can get past that embedding layer, before it can get put in,
put into any dense neurons. Before we can even really do any math with it, we need to turn it into
numbers, right our textual data. 

So now that we know that it's time to talk about RNN , now "RNN are the type of networks we use when we process
textutal data".
Typically, you don't always have to use these, but they are just the best for natural language processing. And that's
why they're kind of their own class. Right, Now, the fundamental difference between a recurrent neural network and 
something like a dense neural network or a convolutional neural network is the fact that it contains an internal loop.

Now what this really means is that the recurrent neural network does not process our entire data at once, so it 
doesnot process the entire training example or the entire input to the model at once. What it does is process it
at different time steps, and maintains what we call an internal memory, and kind of an internal state so that when 
it looks at a new input it will remember what it seen previously, and treat that input based on kind of the context
of the understanding , it's already developed.

Now I understand that this doesnot make any sense right now. But with a dense neural network, or the neural networks
we looked at so far, we call those something called feed forward neural networks. What tha means is we give all of our 
data to it at once, and we pass that data from left to write, I guess for you guys from left to right, so we give
all of the information, you know, we would pass those throught the convolutional layer to start, maybe we would pass
them through dense neurons, but they get given all of the info. 

And then that information gets translated through the network to the very end again, from left to right, Whereas here,
with recurrent neural networks, we actually have a loop which means that we donot feed the entire textual data at once
we actually feed one word at a time, it processes that word, generate some output based on that word, and uses the internal 
memory state that is keeping thte track of to do that as part of the calculation.

So essentially, the reason we do this is because just like " humans, when we, you know, look at text, and we don't just take a
photo of this text and process it all at once we read it left to right, word to word. And based on the words that we've already
read, we start to slowly develop an understanding of what we're reading, right?"
If I just read the word Now,that doesnot mean much to me, If I just read the word encode that doesnot mean much. Whereas, If I 
read the entire sentence, now that we've learned a little bit about how we can encode text, I start to develop an understanding
about what this next word means based on the previous words before it right. 

And that's kind of the point here is that this is what a recurrent neural network is going to do for us, it's going to read one word 
at a time, and slowly start building up its understanding of what the entire data means, and this works in kind of more complicated 
since then, that will draw it out a little bit. But this is kind of what would happen if we um, I guess unraveled a recurrent layer because
recurrent neural networks, yes , it has a loop in it. 

But really, the recurrent aspect of neural network is the layer that implements this recurrent functionality with a loop. Essentially,
what we can see here is that we're saying x is our input, and h is our output, x t is going to be our input at a time t, whereas each T is
going to be our output at time t, if we had a text of, say, length four, so for words like we've encoded them into integers, now at this point
, the first input at time zero will be the first word into our network right, or the first word that this layer is going to see.

And the output at that time is going to be our current understanding of the entire text after looking at just that one word.
Next, what we are going to do is process input one. which will be thte next word in the sentence. But we are going to use the output
from the previous kind of computation of the previous iteration. To do this, so we are going to process this word in combination with what 
we have already seen, and then have a new output, which hopefully should now give us an understanding of what those two words mean.

Next, we'll go to the third word, and so forth and slowly start building our understanding of what the entire textual data forth and slowly start
building our understanding of what the entire textual data means by building it up one by one. The reason we don't pass the entire sequence at
once is because it's very, very diffiult to just kind of look at this huge blob of integeres and figure out what the entire thing means.
If we can do it ony by one and understand the meaning of specific words based on the words that cam before it and start learning those patterns.
That's going to be a lot easier for a neural network to deal with, than just passing it all at once looking at it and trying to get some output.
And that's why we have these recurrent layers. 


"""

import numpy as np
from numpy import random
import seaborn as sns
import matplotlib.pyplot as plt

# Summation vs Addition - n number of addition is summation whereas addition is between any two or limited number of arguments
arr1 = np.array([2, 3, 5])
arr2 = np.array([3, 4, 6])
addition = np.add(arr1, arr2)
products = np.prod([arr1, arr2])
summation = np.sum([arr1, arr2])

print("additions is :", addition)
print("summation is ",summation)
print("product is ", products)

# summation over an x-axis = 1 will sum only over an each array elements
newarr = np.sum([arr1, arr2], axis=1)
proarr = np.prod([arr1, arr2], axis=1)
print("Axis summation over x is : ", newarr)
print("Axuis product is: ", proarr)

# Cummulative Sum means adding partially similar to the fibonacci series using cumsum()
# E.g. The partial sum of [1, 2, 3, 4] would be [1, 1+2, 1+2+3, 1+2+3+4] = [1, 3, 6, 10].
# E.g. The partial product of [1, 2, 3, 4] is [1, 1*2, 1*2*3, 1*2*3*4] = [1, 2, 6, 24]

cummulative = np.cumsum([1,2,3,4])
cum_prod = np.cumprod([1, 2, 3, 4])
print("cummuative is: ", cummulative)
print("cum product is: ", cum_prod)

# Generate linear data
# np.linspace(start, stop, num)
x_linear = np.linspace(-10, 10, 100)
y_linear = 2 * x_linear + 3  # Example linear relationship: y = 2x + 3

# Apply sigmoid function to linear data
y_sigmoid = 1 / (1 + np.exp(-y_linear))

# Plot the linear data and sigmoid-transformed data
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(x_linear, y_linear)
plt.title('Linear Data')
plt.xlabel('x')
plt.ylabel('y = 2x + 3')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x_linear, y_sigmoid)
plt.title('Sigmoid Transformed Data')
plt.xlabel('x')
plt.ylabel('sigmoid(2x + 3)')
plt.grid(True)
plt.tight_layout()
plt.show()