import numpy as np

# a is a (2, 2, 2)
a = np.array(range(8)).reshape((2, 2, 2))
print(a)
print('---------------------------------------')

b = np.array(range(2, 10)).reshape((2, 2, 2))
print(b)
print('---------------------------------------')

print(np.square(a - b))
