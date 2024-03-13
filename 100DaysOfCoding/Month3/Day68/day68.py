

""""
Diagrams:
https://miro.com/app/board/uXjVNhgCCxA=/?share_link_id=173124021394

Step 1. For the green part, we use First Filters will try to find the
dot product between green and first filter box. Numeric values 
obtained after multiplying each of their element's wise numeric 
values inside the boxes.
A. B --> output role of the dot product is a numeric value 
that tells how similar these two originals and the first filter 
box, or the sample and the filter box are.

Step 2. After first green part we move to another grid area greeny 3 x 3 
and compare each pixel with the filters oh we got some similarities 
in the second row second column ie. row2col2
but no filter in the first row second column in the original so we 
set the absence pixel field to 0.

Assign randomly.
numeric values 1 are perfect match and values nearer to 1 are similar then closer to 0.
These features inform us the presence of features in this original map.

we will have tons of different layers as we are constantly expanding, as we go to the depth of the neural network, need lot of computation which leads more time consumption.

---- Rough Idea on How CNN works? -------
STEP 3:
we have generated output feature map from the original image.
Now the next convolutional layer will find the feature except
the output feature maps that means next convolutional layer
will process or find the combinations of lines and edges and
maybe find what a curve is. we slowly work our way up from very,
very small amount of pixels, to finding more and more, almost, 
small amount of pixels, to finding more and more, almost abstract 
different features that exist in the image. And this really allows 
us to do some amazing things with the convolutional neural network.
when we have a ton of differnt layers stacking up on  eachother 
we can pick out all the small little edges which are pretty easy 
to find. And with all these combinations of layers working together,
we can even find things like eyes, feets, or heads

when we have a ton of differnt layers stacking up on 
eachother we can pick out all the small little edges
which are pretty easy to find. And with all these
combinations of layers working together, we can even 
find things like eyes, feets, or heads.

We can find very complicated structures, because we slowly
our work way up starting very easy problem, which are likely 
finding lines, and then finding the combination of lines, edges,
shapes and the very abstract things. 
That's how convolutional neural networks .

"""

"""
------ Padding -------
- Padding makes best sense ways making space, sometimes we want to
make sure that the output feature map from our original image here 
is the same shape or N X N Size and the shape.
In the diagram what we are doing is ewe have original image size is
5 x 5 and the feature image is 3 x 3 so for that if we want to make 
this 5 x 5 as an output what we need to do is simply add the padding
our original image.

We add extra col and row or border around the original image,
adding so must make the pixel at the center of the image as shown
in the diagram.
Observe the red colored has pixel exactly at the center of the 
box.
but the green part has not and they cannot be center with the padding
allows us to do is generate an output map that is the same size as
our original input, and allows us to look at features that are 
maybe right on the edges of images that we might not have been 
able to see. Although this is not important for large but understanding
is and applying for small is fine. X are added to mark cross.

--- STRIDE ----- 
Stride is something that explain how many times we move the sample box
every time that we are about to move it.

Stride of One - Note we have added padding now we move to another pixel
or 1 pixel that is called stride of one.
But we can also take n stride or n times moving or we can two stride
obviously for larger stride then it will be 2 or 4 stride.

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import random

"""
We can use simple arithmetic operations with arithemtic operators
*,+,-,/ directly between numpy arrays 

we can also add condition to the arithmetic operation
but we should use WHERE Clause

"""

arr1 = np.array([10, 11, 12, 13, 14, 15])
arr2 = np.array([20, 21, 22, 23, 24, 25])

result = np.add(arr1, arr2)
print("addition is -->", result)

subtract_r = np.subtract(arr1, arr2)
print("subtraction is -->", subtract_r)

mul_r = np.multiply(arr1, arr2)
print("multiply is -->", mul_r)

div_r = np.divide(arr1, arr2)
print("division is -->", div_r)

pow_r = np.power(arr1, arr2) # arr1 power arr2
print("power is -->", pow_r)

remainder_r = np.mod(arr1, arr2) # arr1 % arr2
print("remainder is -->", remainder_r)
print("mod is -->", remainder_r)

quot_mode = np.divmod(arr1, arr2)
print("quotient and mod ", quot_mode)

arr = np.array([-1, -2, 1, 2, 3, -4])
abs_r = np.absolute(arr)
print("absolute array ", abs_r)

# Exponential Distribution  used for describing time till next event e.g. failure/success etc.
x = random.exponential(scale=2, size=(2, 3))
print("exponent-->", x)

sns.distplot(random.exponential(size=9000), hist=False, color="r")
plt.title("Exponential Distribution ")
plt.show()


"""
In probability theory and statistics, the exponential distribution is a continuous probability distribution
that describes the time between events in a Poisson process, where events occur continuously
and independently at a constant average rate. The scale parameter, often denoted as λ (lambda), determines the rate at which events occur.
In the context of the code you provided, the scale parameter is set to 2. This means that 
the average rate of occurrence of events (or the mean) in the exponential distribution 
is 1/2, or 0.5 events per unit of time (since the scale parameter is the reciprocal of 
the rate parameter λ).

Now, let's interpret the output:

Each element in the 2x3 array represents a random sample drawn from the 
exponential distribution with a scale parameter of 2. Here's a breakdown 
of the interpretation:

The first row [0.6569592 0.15254679 1.22530918] represents three 
random samples from the exponential distribution.

The second row [2.90125227 2.86376289 1.31681472] represents another 
three random samples from the same distribution.
The numbers themselves represent the time intervals between successive events. 
For example, the first element in the first row (0.6569592) may represent 
the time between two consecutive events occurring in a process with an 
average rate of 0.5 events per unit of time, which in this case, is 
about 0.6569592 units of time.

"""