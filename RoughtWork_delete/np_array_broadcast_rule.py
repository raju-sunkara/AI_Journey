import numpy as np

# Define two 2-dimensional arrays
array1 = np.array([[1, 2], 
                   [3, 4]])
array2 = np.array([[5, 6], 
                   [7, 8]])

# Add the arrays
result = np.add(array1, array2)

allSum = np.sum(array1,axis=None)
print("Entire Sum = ",allSum)

rowSum = np.sum(array1,axis=1, keepdims=True)
print("Row Sum = ",rowSum)

rowSum = np.sum(array1,axis=1, keepdims=False)
print("Row Sum = ",rowSum)

colSum = np.sum(array1,axis=0)
print("Column Sum = ",colSum)

# Print the result
print("Array 1:")
print(array1)
print("Array 2:")
print(array2)
print("Summation of Array 1 and Array 2:")
print(result)