import numpy as np

o1 = np.array([
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11]], dtype=np.float32)
o2 = np.array([
    [0, 2, 4, 6],
    [8, 10, 12, 14],
    [16, 18, 20, 22]], dtype=np.float32)
original = np.dstack((o1, o2))
print(original.shape)
# print(original[:, :, 1])

d1 = np.zeros((3, 4), dtype=np.float32)
d2 = np.ones((3, 4), dtype=np.float32)
demosaic = np.dstack((d1, d2))
print(demosaic.shape)
# print(demosaic[:, :, 1])

# diff = original - demosaic
# print(diff.shape)
# print('diff[:, :, 0]:\n', diff[:, :, 0])
# print('diff[:, :, 1]:\n', diff[:, :, 1])
#
# diff_sum = np.sum(diff, axis=2)
# print('diff_sum:\n', diff_sum)

np.set_printoptions(precision=3)

diff = np.sqrt(np.sum(np.square(original - demosaic), axis=2))
print(diff)

