import numpy as np

def increment(aa, li):
    li.append(1)
    li.append(2)

    for i in range(len(aa)):
        aa[i] = 0



a = np.array([1, 2 ,3])
l = list()

print(a)
print(l)
increment(a, l)
print(a)
print(l)
