

import numpy as np

# filtering the array
arr = np.array([12, 45, 67, 89])
# we can use either boolean values True or False to filter out
# usually filter does is create a new array by filtering the required elements from array
x = ([True, False, True, True])
print(arr[x])
# [12 67 89]

# in the above example we hard coded but in real dynamic scence we do it by condition in looping

# creating a empty list
input = 45
filtered_list = []
for ele in arr:
    if ele > input:
        filtered_list.append(True)
    else:
        filtered_list.append(False)
        
newarr = arr[filtered_list]
print("filterd list", newarr)
# output filterd list [67 89]