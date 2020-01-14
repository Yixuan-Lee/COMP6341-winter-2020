import cv2 as cv
import numpy as np

# ----- Loading images and basics -----

img = cv.imread("sample.jpg")
cv.imshow("Hey", img)
cv.waitKey(0)  # opencv waits

# x: # of columns
# y: # of rows
# c: B, G, R channel
print(img.shape)  # (x, y, c)

# may help for noise issue
# uint8: [0-255]
# float: [0-1]
print(img.dtype)

# ----- Convert the image to grayscale -----
# RGBA: A means transparency
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Hey", img_gray)
cv.waitKey(0)

# x: # of columns
# y: # of rows
print(img_gray.shape)  # (x, y)

print(img_gray.dtype)  # does not change

# --- Access the RGB pixel located at x=50, y=100, keeping in mind that
#     OpenCV stores images in BGR order rather than RGB ---
# 100th column, 50th row
# 100 is from left to right
# 50 is from top to bottom
b_pixel, g_pixel, r_pixel = img[100, 50]
print("B={}, G={}, R={}".format(b_pixel, g_pixel, r_pixel))

# 100-150 columns
# 200-250 rows
roi = img[100:150, 200:250]
cv.imshow("JAasa", roi)
cv.waitKey(0)

# --- Numpy is an optimized library for fast array calculation ---

# accessing RED value (better way than img[10, 10, 2] for assignment 2)
print(img.item(10, 10, 2))

# modifying RED value
img.itemset((10, 10, 2), 100)
print(img.item(10, 10, 2))

# clipping due to overflow
x = np.uint8([250])
y = np.uint8([10])
print(cv.add(x, y)) # good : 250+10 = 260 => 255
print(x + y)        # bad  : 250+10 = 260 % 256 = 4


# --- Filters in OpenCV ---
# the filter below:
#   0  -1   0
#   -1  5   0
#   0  -1   0
# will sharpen the image
# create a kernel
kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]], np.float32)  # np.float32 is common for kernel dtype
# -1 is for image depth, -1 means input and output have the same # of channels
img_mod = cv.filter2D(img, -1, kernel)
cv.imshow("sharpened", img_mod)
cv.waitKey(0)

# apply a box filter
box_kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]], np.float32)  # np.float32 is common for kernel dtype
img_box_kernel = cv.filter2D(img, -1, box_kernel)   # wrong way (every pixel tends to white)
cv.imshow("With box filter (wrong)", img_box_kernel)
cv.waitKey(0)
img_box_kernel = cv.filter2D(img, -1, box_kernel/9) # right way
cv.imshow("With box filter", img_box_kernel)
cv.waitKey(0)


# --- Gaussian blur and box blur ---
blank_image = np.zeros((100, 100))

# 0 is the variance
cv.GaussianBlur(img, (3, 3), 0, img)
cv.imshow("Gaussian Blurred", img)
cv.waitKey(0)

