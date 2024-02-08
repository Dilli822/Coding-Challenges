import tensorflow as tf

# Creating tensors using tf.constant
# since tensor is a fundamental data structure in neural network
# 0 dimension or 0 rank/degree tensor

string = tf.constant("this is a string", tf.string)
number = tf.constant(224, tf.int16)
floating = tf.constant(2.577, tf.float64)

print("String tensor:", string)
print("Number tensor:", number)
print("Floating-point tensor:", floating)

# output of 
# String tensor: tf.Tensor(b'this is a string', shape=(), dtype=string)
# Number tensor: tf.Tensor(224, shape=(), dtype=int16)
# Floating-point tensor: tf.Tensor(2.577, shape=(), dtype=float64)


# 1 dimension tensor or rank/degree 1 tensor
# just adding one array can make it 1-D 

rank1_tensor = tf.Variable(["this is a string"], tf.string)
rank2_tensor = tf.Variable( [ ["array 1", "thursday"],["array 2", "friday"] ], tf.string)
rank3_tensor = tf.Variable( [ ["array 1", "thursday", "thursday"],["array 2", "friday","thursday"]],[["array 2", "friday","thursday"] ], tf.string)

# find the rank
tf.rank(rank1_tensor)
tf.rank(rank2_tensor)

# rank2_tensor has multi list two sublists inside the single parent list and thus it is a matrix

print("tf.rank of rank1 is ---->", tf.rank(rank1_tensor))
print("tf.rank of rank2 is ---->", tf.rank(rank2_tensor))
print("tf.rank of rank3 is ---->", tf.rank(rank3_tensor))



# find the shape of a tensor
tf.shape(rank1_tensor)

# shape --> tells how many elements are inside in each dimension
#       ---> it tells the shape of tensor but sometimes, it may be unknown 

print(tf.shape(rank1_tensor))  # tf.Tensor([1], shape=(1,), dtype=int32)
print(tf.shape(rank2_tensor))  # tf.Tensor([2 2], shape=(2,), dtype=int32) # since we have two elements in second Dimension
print(tf.shape(rank3_tensor))  # tf.Tensor([2 3], shape=(2,), dtype=int32) # since we have three elements in each second Dimension




# Ref read more about tensorflow
# https://en.wikipedia.org/wiki/TensorFlow