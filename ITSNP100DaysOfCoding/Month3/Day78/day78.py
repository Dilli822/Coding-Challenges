
"""
Previously:
We didnot keep ordering and sequence of the words, we directly keep it
in bag of words and keep the track of frequency of words.
That will lose the ordering and sequencing whole encoding the textual, 
context and may be meaning sometimes.

1. Technique 1 - Context of Words
Instead we encode whole text in its original form including space
and assign integers and later encrypt it.

If sentence "Gurkha is a brave warrior."
[0, 1, 2]
[2, 1, 0] -> this could be brave warrior are gurkhas.
Above two are not same unlike in bag of words.

Flaw of these array type methods/Flaws
Suppose we have 100, 000 words each word have unique mappings or integer
associated with it.

1: happy
2: sad

100,000: good
as humans by kind of just thinking about 
let's consider the fact that we 're gonna try to classify sentences
as a positive or negative so sentiment analysis the words happy and
good in that regard you know sentiment analysis are probably pretty
similar words right and then if we were going to group these words we'd
probably put them in a similar group we'd classify them as similar 
words we could probably interchange them in a sentence and it wouldn't
change the meaning a whole ton I mean it might but you might not as well
and that we could say these are kind of similar but our model or our
encoder right whatever we are doing to translate our text into integers
here has decided that a hundred thousand is gonna represent good and 
one is gonna represent happy and well there's an issue with that because
that meand when we pass in something like one or a hundred thousand to 
our model it's gonna have a very difficult time determining the fact that
one and a hundred thousand although they are 99,999 connect units apart
are actually very similar words and that's the issue we get into when 
we do something like this is that numbers we decide to pick to represent
each word are very important and we don't really a way of being able to
look at words group them and saying okay well we need to put all of the 
happy words in the range like 0 to 100 all of the like adjectives in this
range we don't really have a wayt to do that and this gets even harder
for our model when we have these arbitary mappings right and then we
have something like 2 in between where 2 is very close to 1 right 
yet these words are complete opposites in fact I'd say they're probably
polar opposites our model trying to that the differnce between 1 & 2
is actually way larger  than the difference between one and a hundred 
thousand is gonna be very difficult and say it's even able to do that 
as soon as we throw in the mapping 99,999: bad. well now it gets even
more difficult because it's now like okay what the range is this big then
that means these words are actually very similar but then you throw another
word in here like this and it messes up the entire system so that's kind
of what I wanted to show is that that's where this breaks apart on these
vocabularies and that's why I'm going to introduce us now to another 
concept called word embedings.

"""

"""
Word Embeddings:
What it does is essentially

try to find a way to represent words
that are similar using very similar numbers and in fact what a word
embedding is actually gonna do, I'll talk about this more in detail
as we go on is classify or translate evey single one of our words
into a vector and that vector is gonna have some you know "n" amount of 
dimensions usually we are gonna use something like 64 maybe 128 dimensions
for each vector and every single component of that vector will kind of 
tell us what group it belongs to or how similar it is to other words so 
we're gonna create something called the world embeddings now don't ask why
it's called embeddings i don't know the exact reason but i believe it's to 
have has to do something with the fact that they're vectors and let's
just say we have 3d plane like this so we've already kind of looked at what
vectors are before so.

I'll skip over explaining them and what we are gonna do is take some word
so let's say we have the word "good" and instead of picking some integer
represent it we are gonna pick some vector which means we are gonna
draw some vector in this 3d space actually. let's make this a different 
color red. this red vector like this and this vector represents this
word good and in this case we'll say we have x1, x2, x3 as dimensions
that means every single word in our data set will be represented by
three co-ordinates so one vector with three different dimensions where
we have x1, x2 and x3 and our hope is that by using this word embeddings
layer and 

we'll talk about how it accomplished this in a second is that 
we can have vectors that represents very similar words being very similar
which means that you know if we have the vector "good" here we would hope
the vector happy from our previous example right would be " a vector that 
points in a similar direction to it " that is kind of a similar looking 
thing where the angle between these two vectors right and maybe I'll draw
it here so we can see is small so that we know that these words are similar
and then we would hope that if we had a word that was much different maybe 
say like the word "bad" would " point in a different direction the vector "
that represents it and that would tell our model because the angle between
these two vectors are so big that these are veru different words right

now in theory does the embedding work layer work like this you know not 
always but this is what it's trying to do is " essentially pick some 
representation in a vector form for each word and then " these vectors
we hope if they're similar words are going to be pointing in a very similar
direction " and that's kind of the best explanation of the word embeddings
layer I can give you how do we do this though? how do we actually you know
go from word to vector and have that be meaningful well this is actually

we call a LAYER so Word Embeddings is actually a layer and it's 
something we're going to add to our model and that means that this actually
learns the embeddings for our words and the way it does that is by trying 
to kind of pick out context in the sentence and determine based on where
a word is in sentence kind of what it means and that encodes it doing
that now I know that's kind of a rough explanation to give to you guys.

I don't want to go too far into word embeddings in terms of the math
because I donot want to get you know waste our time or get too 
complicated if we don't need to but just understand that our word
embeddings are actually trained and that the model actually learns these
word embeddings as it goes and we hope that by the time it's looked at 
enough training data.

It's determined really good ways to represent all
of our different words so that they make sense to our model in the 
further layers and we can use pre-trained word embedding layers if we like
just like we use that pre-trained Convolutional Base in the previous
section and we might actually end up doing that actually probably not
in this tutorial but it is something to consider that you can't do that so
that's how word embeddings work this is how we encode textual data and 

this is why it's so important that we kind of consider the way that we pass
information to our neural network because it makes a huge differences.

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import log

arr = np.arange(1, 10) # 10 not included 

log_r = np.log2(arr)
log_10_r = np.log10(arr)
log_er = np.log(arr)

# Define a function to calculate logarithm at any base for an array of numbers
def nplog_array(numbers, base):
    return [log(x, base) for x in numbers]

# Calculate logarithm at any base for the entire array 'arr' with base 15
log_any_base = nplog_array(arr, 15)

print("arr :", arr)
print("log2 of arr: ", log_r)
print("log10 of arr: ", log_10_r)
print("log at e of arr: ", log_er)
print("log at any base of arr (base 15): ", log_any_base)

# Plot using Seaborn
sns.set()  # Set Seaborn styles
plt.figure(figsize=(8, 6))  # Set figure size

# Plot the base-2 logarithm
sns.lineplot(x=arr, y=log_r, label='log2')

# Plot the base-10 logarithm
sns.lineplot(x=arr, y=log_10_r, label='log10')

# Plot the natural logarithm (base-e)
sns.lineplot(x=arr, y=log_er, label='log at e')

# log at any base for that we must use python in-built frompyfunc() that takes 2 input and 1 output 
# In this case, it's 2, indicating that the log function will receive two arguments: the number and the base.
# Plot the logarithm at any base (base 15)
sns.lineplot(x=arr, y=log_any_base, label='log at any base (15)')

# Plot customization
plt.title('Logarithmic Plot')
plt.xlabel('Number')
plt.ylabel('Logarithm')
plt.legend()

# Show plot
plt.show()
