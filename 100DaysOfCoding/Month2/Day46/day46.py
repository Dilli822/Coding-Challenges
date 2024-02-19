import numpy as np
import matplotlib.pyplot as plt


# Below is a list of all data types in NumPy and the characters used to represent them.

# i - integer
# b - boolean
# u - unsigned integer
# f - float
# c - complex float
# m - timedelta
# M - datetime
# O - object
# S - string
# U - unicode string
# V - fixed chunk of memory for other type ( void )

# dtype returns the data types
array_int = np.array([1, 3, 5, 7, 9])
print(array_int.dtype)

array_string = np.array([ "hello", "world", "i am", "dilli"])
print(array_string.dtype)

# Creating Arrays With a Defined Data Type
array_withdefineddtype= np.array([1,2,3,4], dtype = "S")
print(array_withdefineddtype)
print(array_withdefineddtype.dtype)

# 1 byte = 8bits and 4 byte = 32bits
# array data types with data types 4 byte integer
array_with_data_4byte_int = np.array([1,2,3,4], dtype="i4")
print(array_with_data_4byte_int)
print(array_with_data_4byte_int.dtype)

# What if a Value Can Not Be Converted? --> raise error example of raising error
# array_eg_error_conversion = np.array([ "x","2",3,4], dtype='i')
# print(array_eg_error_conversion) # ValueError: invalid literal for int() with base 10: 'x'


# Converting Data Type on Existing Arrays
# best way is to copy the existing array to new dtype
arr_str = np.array(["1", "2", "3"])
new_int_array = arr_str.astype("i")
print(new_int_array)
print(new_int_array.dtype)


# integer to boolean value
arr_inte = np.array([0, 2, 4, 6, 8])
new_bool_arr = arr_inte.astype(bool)  # except 0 all integer are True
print(new_bool_arr)
print(new_bool_arr.dtype)