# read more docs on https://docs.google.com/document/d/12XKx4-R0_YSgdCQT0q3y8LJn87hWVHlFkWOffCIVdJQ/edit?usp=sharing


# read more docs on https://docs.google.com/document/d/12XKx4-R0_YSgdCQT0q3y8LJn87hWVHlFkWOffCIVdJQ/edit?usp=sharing

# 
def linear_regression(X, y):
    n = len(X)
    sum_x = sum(X)
    sum_y = sum(y)
    sum_xy = sum(x * y for x, y in zip(X, y))
    sum_x_squared = sum(x ** 2 for x in X)

    # Calculate coefficients (a and b)
    a = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
    b = (sum_y - a * sum_x) / n

    return a, b

# Example usage:
X = [1, 2, 3, 4, 5]
y = [2, 3, 4, 5, 6]

a, b = linear_regression(X, y)
print("Coefficient (a):", a)
print("Intercept (b):", b)


# numbers = [1, 2, 3, 4, 5]
# total = sum(numbers)
# print(total)  # Output: 15

# X = [1, 2, 3, 4, 5]
# y = [2, 3, 4, 5, 6]

# # Using zip to combine X and y
# combined = zip(X, y)
# print(list(combined))  # Output: [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
