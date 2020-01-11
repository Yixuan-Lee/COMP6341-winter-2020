import numpy as np

arr = np.array([[1.0, 2], [3, 4]])
print(arr.shape)
brr = np.expand_dims(arr, axis=2)
print(brr.shape)

print(arr)
print(brr)

print(arr.dtype)
arr = arr.astype('float32')
print(arr.dtype)


