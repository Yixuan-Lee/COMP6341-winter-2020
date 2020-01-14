""" Exercise:
Create a 100 x 100 image
Display it

Make the absolute center pixel white
Imshow it

Run a box filter kernel through it using filter2d function or
gaussian blur
Imshow it again and compare
"""

import cv2 as cv
import numpy as np

image = np.zeros((500, 500))
cv.imshow("initial image", image)
cv.waitKey(0)

# print(image.shape)
# print(image.shape[0] / 2)
half = image.shape[0] / 2
for i in range(200, 300):
    for j in range(200, 300):
        image.itemset((i, j), 255)
# image.itemset((10:40, 20:50), 255)

cv.imshow("modified image", image)
cv.waitKey(0)

# box filter
# box_filter = np.ones((20, 20))
# image_box = cv.filter2D(image, -1, box_filter / 400)
# cv.imshow("With box filter", image_box)
# cv.waitKey(0)
# expected result: the white square gets bigger

# gaussian filter
out = np.empty(image.shape)
cv.GaussianBlur(src=image, ksize=(5, 5), sigmaX=0, dst=out)
cv.imshow("With gaussian filter", out)
cv.waitKey(0)

