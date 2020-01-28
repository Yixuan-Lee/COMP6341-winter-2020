import cv2
import numpy as np

# 1. Load an image and creating the kernels
img = cv2.imread('../images/oldwell.jpg', cv2.IMREAD_GRAYSCALE)

'''
# used to detect vertical edges
sobel_X = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]])
# used to detect horizontal edges
sobel_Y = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]])

# 2. Applying filters to get the derivatives
dx = cv2.filter2D(img, -1, sobel_X)
dy = cv2.filter2D(img, -1, sobel_Y)
# add the two components together
sobel_img = dx + dy

# 3. Displaying results
cv2.imshow('image', img)
cv2.imshow('DX', dx)
cv2.imshow('DY', dy)
cv2.imshow('Sobel', sobel_img.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

# 4. Repeat the same with the built-in sobel operator (ksize should be 3)
sobelx = cv2.Sobel(img, -1, 1, 0, ksize=3)
sobely = cv2.Sobel(img, -1, 0, 1, ksize=3)

sobel_final = cv2.add(sobelx, sobely)

# 5. Displaying the results
cv2.imshow('Sobel X', sobelx)
cv2.imshow('Sobel Y', sobely)
cv2.imshow('Sobel Result', sobel_final)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 6. Canny Edge Detector
# min threshold = 100
# max threshold = 200
# assignment-2 : implement cv2.Canny function
canny_edges = cv2.Canny(img, 100, 200) # play around with the values

cv2.imshow('Canny Edges', canny_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

# 7. Gaussian Pyramid
image_clone = img.copy()
g_pyramid = [image_clone]

for i in range(3):
    # before downsampling (drop information), we apply a Gaussian filtering
    G = cv2.pyrDown(g_pyramid[i]) # notice indexing
    cv2.imshow('Down Image ' + str(i+1), G)
    g_pyramid.append(G)
    cv2.waitKey(0)
cv2.destroyAllWindows()

# 8. Upsampling the downsampled images
upsampled = g_pyramid[-1] # last element

for i in range(3):
    # doing gaussian blur before upsampling
    upsampled = cv2.pyrUp(upsampled)
    cv2.imshow("Up Image " + str(i), upsampled)
    cv2.waitKey(0)
cv2.destroyAllWindows()

# 9. use Laplacian pyramid from the upsampled and downsampled images
l_pyramid = [g_pyramid[3]]

for i in range(3, 0, -1):
    # dstsize is not mandatory
    #   g_pyramid[i-1].shape[1]: number of rows
    #   g_pyramid[i-1].shape[0]: number of columns
    GE = cv2.pyrUp(g_pyramid[i], dstsize = (g_pyramid[i-1].shape[1], g_pyramid[i-1].shape[0]))
    L = cv2.subtract(g_pyramid[i-1], GE)
    # give the differences between the original and the zoom-in-ed downsampled image
    cv2.imshow('L Image ' + str(i), L)
    cv2.waitKey(0)
    l_pyramid.append(L)
cv2.destroyAllWindows()


# 10. recreate the original image using Laplacian and the zoom-in-ed downsampled image
ls = l_pyramid[0]
for i in range(1, 4):
    ls = cv2.pyrUp(ls, dstsize= (l_pyramid[i].shape[1], l_pyramid[i].shape[0]))
    ls = cv2.add(ls, l_pyramid[i])
    cv2.imshow('Up Laplacian Image ' + str(i), ls)
    cv2.waitKey(0)
cv2.destroyAllWindows()


# 11. (Optional) Blending: merge 2 images and make it as realistic as possible
# 2 images have to be the same size, or resize them to the same size
#
# create 3 Gaussian pyramids (GPs) :
#   2 GPs for 2 images
#   1 GP for the combined image
apple = cv2.imread('../images/apple.jpg')
orange = cv2.imread('../images/orange.jpg')
apple_orange = np.hstack((apple[:, :154], orange[:, 154:]))


# zip(): a list of pairs

