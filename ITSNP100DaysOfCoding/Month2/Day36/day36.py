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


# it must have uniform or equal number of elements inside the array 

rank_tensor = tf.Variable( [ ["array 1", "thursday", "thursday"],["array 2", "friday","thursday"]],[["array 2", "friday","thursday"] ], tf.string)
print(rank_tensor.shape) # give 2, 3 rank 2 with 3 elements


rank_tensor = tf.Variable( [["array 1", "thursday", "thursday"],["array 2", "friday","thursday"],["array 2", "friday","thursday"] ], tf.string)
print(rank_tensor.shape) # give 3, 3 rank 3 with 3 elements



# Tensor Addition
tensorA = tf.constant([[1, 2], [3, 4], [5, 6]])
tensorB = tf.constant([[1, -1], [2, -2], [3, -3]])

# Tensor Addition
tensorNew = tf.add(tensorA, tensorB)
print(tensorNew)
# Result: [ [2, 1], [5, 2], [8, 3] 

# Tensor subtraction
tensorNew = tf.sub(tensorA, tensorB)
print(tensorNew)
# Result: [ [2, 1], [5, 2], [8, 3] ]

# Tensor Multiplication
tensorNew = tf.multiply(tensorA, tensorB)

# Print the result [4 8 6 8]

print(tensorNew.numpy())

tensorNew = tensorA.div(tensorB);
print(tensorNew.numpy())
# // Result: [ 2, 2, 3, 4 ]

tensorNew = tensorA.square();
print(tensorNew.numpy())
# // Result [ 1, 4, 9, 16 ]


tensor = tf.constant([[1, 2, 3],
                      [4, 5, 6]])

# Reshape the tensor
reshaped_tensor = tf.reshape(tensor, [3, 2])

# Print original and reshaped tensors
print("Original tensor:")
print(tensor.numpy())
print("Reshaped tensor:")
print(reshaped_tensor.numpy())

# Original tensor:
# [[1 2 3]
#  [4 5 6]]
# Reshaped tensor:
# [[1 2]
#  [3 4]
#  [5 6]]


import tensorflow as tf

# Define constants
a = tf.constant(5)
b = tf.constant(2)

# Addition
addition = tf.add(a, b)  # Equivalent to a + b

# Subtraction
subtraction = tf.subtract(a, b)  # Equivalent to a - b

# Multiplication
multiplication = tf.multiply(a, b)  # Equivalent to a * b

# Division
division = tf.divide(a, b)  # Equivalent to a / b

# Exponentiation
exponentiation = tf.pow(a, b)  # Equivalent to a ** b

# Modulo
modulo = tf.mod(a, b)  # Equivalent to a % b

# Absolute value
absolute_value = tf.abs(tf.constant([-3, -2, -1, 0, 1, 2, 3]))

# Print results
with tf.Session() as sess:
    result_add, result_sub, result_mul, result_div, result_exp, result_mod, result_abs = sess.run(
        [addition, subtraction, multiplication, division, exponentiation, modulo, absolute_value])
    print("Addition:", result_add)
    print("Subtraction:", result_sub)
    print("Multiplication:", result_mul)
    print("Division:", result_div)
    print("Exponentiation:", result_exp)
    print("Modulo:", result_mod)
    print("Absolute value:", result_abs)


# Addition: 7
# Subtraction: 3
# Multiplication: 10
# Division: 2.5
# Exponentiation: 25
# Modulo: 1
# Absolute value: [3 2 1 0 1 2 3]

# The with tf.Session() as sess: statement creates a TensorFlow session named sess. It ensures that the session is properly closed after executing the code block, even if an exception occurs.

# Inside the session block, sess.run() is used to execute the TensorFlow operations defined earlier (addition, subtraction, etc.). It takes a list of tensors to evaluate (in this case, the results of various mathematical operations) and returns their computed values.

# The results of the operations are assigned to the variables result_add, result_sub, etc., using tuple unpacking.

# Finally, the results are printed using print() statements.



# Ref read more about tensorflow
# https://en.wikipedia.org/wiki/TensorFlow