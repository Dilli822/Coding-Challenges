
# Source: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
"""
LSTM (Long Short-Term Memory):

The layer we discussed in depth above was called a simple RNN. However, there does exist some other recurrent
layers(layers that contain a loop) that work much better than a simple RNN layer. The one we will talk about 
here is called LSTM(Long Short-Term Memory). This layer works very similarity to the SimpleRNN layer but adds
a way to access inputs from any timestep in the past. Whereas in our simple RNN layer input from previous 
timestamps gradually disappeared as we got further through the input. With a LSTM we have a long term memory
data structure storing all of the previously seen inputs as well as when we saw them. This allows for us to 
access any previous value we want at any point in time. This adds to the complexity of our network and allows
it to discouver more useful relationshios between inputs and when they appear.
For the purpose of this course we will refrain from going any further into the math or details behind these 
layers work. 

"""

"""
Continued from 4:52 video:

That's going to be a lot easier for a neural network to deal with, than just passing it all at once looking at
it and trying to get some output.And that's why we have these recurrent layers. 
And I'm going to go through them and then we'll talk a little bit more in depth of how they work. So the first
one is called " long short term memory" and actually in fact, before we get into. let's uh, talk about just a 
first, like a simple layer so that we kind of have a reference point before going here. Okay, so this is kind
of the example I want to use here to illustrate how a recurrent network works and a more.

Teaching style ranther than what I was doing before. So essentially, the way that this works is that this whole thing 
that I'm drawing here, right, all of this circle stuff is really "one layer", And what I'm doing right now is breaking 
this layer apart and showing you kind of how this works in a series of steps. So rather than passing all the information
at once, we're going to pass it as a sequence, which means that we're going to have all these different words
and we're going to pass them one at a time to the kind of to the layer, right to this recurrent layer. So we're
going to start from this left side over here, as well as right, you know, start over here at time step zero.
That's what zero mean. so time step is just you know the order. 

In this case, this is the first word. So let's say we have the sentence Hi, I am Dilli, right, we've broken 
these down into vectors,they've been turned into their numbers, I'm just writing them here. So, we can kind of see 
what I mean like a natural language. And they are the input to this recurrent layer. So all of our different words,right, 
that's how many kind of little cells we're going to draw here is how many words we have in this sequence that we're 
talking about. So in this case, we have four right four words, So that's why I've drawn four cells to illustrate that.
Now, what we do is that time step zero, the internal state of this layer is nothing, there's no previous output, 
we haven't seen anything yet. 

Which means that this first kind of cell,which is what I'm looking at right here, What I'm drawing
in this first cell is only going to look and consider the first word and kind of make some prediction about it
and do something with it, we're going to pass high to the cell, some math is going to go on in here. And then
what it's going to do is it's going to output some value, which you know tells us something about the word Hi,
right,some numeric value, we're not going to talk about what that is, but it's gonna be there's gonna be some
output. 

Now, what happens is after the cell has finished processing this, so right, so this one's done, this is 
completed at zero, the outputs there, we'll do check mark to say that that's done, it's finished processing,
this output gets fed into actually the same thing. Again, we're kind of just keeping track of it. And now what
we do is we process the next input which is "I".

And we use the output from the previous cell to process this and understand what it means. So now,technically,
we should hace some "output" from the previous cell. So from whatever "HI" was right, we do some analysis 
on the word I, we kind of combine these things together. And that's the output of this cell is our understanding
of not only the current input, but the previous input with the current input. 

So we're slowly kind of building of what this word "I " means based on the words, we saw before. And that's the
point I am trying to get at is that " this network uses what it's seen previously, to understand the next thing
that it sees it's building a context is trying to understand not only the word but what the word means, you know
in relation to what's come before it. So that's what's happening here. So then this output here right, we get
some output, we finish this, we get some output h1, h1 is passed into here And now we have the understanding of
what " HI " & "I" means. And we add them like that, we do some kind of computations, we build an understanding
what the sentences, and then we get the output h2, that passes to h3.

And they'll finally, we have this final output "h3" which is going to understand hopefully, what this entire means
Now this is good, this works fairly well. And this is called  a Simple RNN Layer, which means that all we do is
we take the output from "the previous cell of the previous iteration because really all of these cells is just
an iteration, almost an a for loop, right based on all the different words in our sequence. 

And we slowly start building to that understanding as we go throught the entire sequence. Now, the only issue
with this is that as we have " a very long sequence, so sequences of length, say 100, or 150, the begining of those
sequences starts to kind of get lost as we go through this because remember, all we 're doing right is the output
from 'h2' is really a combination of the output from h0 and h1."
And there's a new word what we've looked at, and h3 is now a combination of everything before it and this new
word. So it becomes increasingly difficult for our model to actually build a really good understanding of the 
text in general when the sequence gets long because it's hard for it to remember what it seen at the very beginning
because that is now so insignificant.

There's been so many outputs tracked on to that. ' It is hard for it to go back and see that if that makes
any sense.'

"""
import numpy as np
import matplotlib.pyplot as plt

# NumPy difference
arr = np.array([20, 25, 35, 40])
newarr = np.diff(arr)
repeat_diff = np.diff(arr, n = 2)

# [25-20, 35-25, 40-35] = [5, 10, 5] , then, since n=2, we will do it once more, with the new result: [5-5, 10-5, 5-10] = [, 5, -5]
print("np diff: ", newarr)
print("np diff twice with n param: ", repeat_diff)

# NumPy finding LCM
a = 45
b = 4
lcm_r = np.lcm(a,b)
print("numpy lcm is :", lcm_r)

# findind lcm in arrays, first we must reduce it
# The reduce() method will use the ufunc, in this case the lcm() function, on each element, and reduce the array by one dimension.
lcm_arr = np.lcm.reduce([2, 4, 8])
lcmMu = np.arange(1, 11)
lcm_arr2 = np.lcm.reduce(lcmMu)
print("lcm in an array: ", lcm_arr)
print("lcm in an array example: ", lcm_arr2)
# LCM(1, 2) = 2
# LCM(2, 3) = 6
# LCM(6, 4) = 12
# LCM(12, 5) = 60
# LCM(60, 6) = 60
# LCM(60, 7) = 420
# LCM(420, 8) = 840
# LCM(840, 9) = 2520
# LCM(2520, 10) = 2520

# Sample data
# Sample data (smaller set)

import matplotlib.pyplot as plt
import numpy as np

# Sample data
hari_days = [2,4,6,8, 10]  # Days Hari (Alice) plays video games
krishna_days = [3, 6, 9, 12]  # Days Krishna (Bob) reads books

# Calculate the intervals between consecutive activities
intervals_hari = np.diff(hari_days)
intervals_krishna = np.diff(krishna_days)

# Find the LCM of the intervals
hari_lcm = np.lcm.reduce(intervals_hari)
krishna_lcm = np.lcm.reduce(intervals_krishna)

# Plotting the timeline
plt.figure(figsize=(10, 5))

# Plot Hari's activity
plt.plot(hari_days, ['Hari']*len(hari_days), 'ro', label='Hari Plays (Video Games)')

# Plot Krishna's activity
plt.plot(krishna_days, ['Krishna']*len(krishna_days), 'bo', label='Krishna Reads (Books)')

# Highlight where both activities coincide (multiples of LCM)
coincide_days = [day for day in range(1, max(hari_days[-1], krishna_days[-1]) + 1) if day % hari_lcm == 0 and day % krishna_lcm == 0]
for day in coincide_days:
    if day in hari_days and day in krishna_days:
        plt.plot(day, 'go', markersize=10, markeredgecolor='black')  # Green dot for collision

# Add text annotations for clarity
plt.title('Activity Timeline for Hari and Krishna')
# Add text annotations for clarity at the top
plt.text(0.5, 1.05, f'Hari\'s LCM: {hari_lcm} means that Hari\'s activity cycle repeats every 2 days.', transform=plt.gca().transAxes, fontsize=10,horizontalalignment='center' )
plt.text(0.5, 1.10, f'Krishna\'s LCM: {krishna_lcm} means that Krishna activity cycle repeats every 3 days.', transform=plt.gca().transAxes, fontsize=10, horizontalalignment='center')

plt.yticks(['Hari', 'Krishna'])
# Original x-axis label for days
plt.xlabel('Days')
# Add a separate text annotation for the total LCM with descriptive text
plt.text(0.5, -0.13, f'Total LCM: {hari_lcm * krishna_lcm} Hari and Krishna activities will coincide.', 
         transform=plt.gca().transAxes, fontsize=10, horizontalalignment='center')
plt.legend()
plt.xticks(range(0, max(hari_days[-1], krishna_days[-1]) + 1, 1))  # Set x-axis ticks with increment of 1 day
plt.grid(True)
plt.show()

