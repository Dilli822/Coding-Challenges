import tensorflow as tf
# Create a constant tensor
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])

# Print the tensor
print("Tensor:")
print(tensor)

# Run a TensorFlow session to evaluate the tensor
with tf.Session() as sess:
    result = sess.run(tensor)
    print("\nResult:")
    print(result)
