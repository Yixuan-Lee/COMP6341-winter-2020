#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[3]:


import cv2
import numpy as np


# The Sobel Operator is a discrete differentiation operator. It computes an approximation of the gradient of an image intensity function. 
# 
# Horizontal changes/Vertical Changes: The horizontal change (gx) is computed by convolving an image (I) with a kernel with odd size. For example for a kernel size of 3, Gx would be computed as:
# 
# <img src = 'imgs/gx.png'>
# 
# Similarly for Gy :
# 
# <img src = 'imgs/gy.png'>
# 
# Then we combine both x and y component to find the result by doing I = gx + gy. Or in other notation I = dx+dy as sobel operator gives us the first derivative.

# ## Load an image and creating the kernels

# In[4]:


img = cv2.imread('imgs/oldwell.jpg',cv2.IMREAD_GRAYSCALE)

sobel_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
sobel_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])


# ## Applying the filters to get the derivatives

# In[5]:


dx = cv2.filter2D(img,-1,sobel_X)
dy = cv2.filter2D(img,-1,sobel_Y)

## Add the two components to get the final result
sobel_img = dx + dy 


# ## Displaying results

# In[6]:


cv2.imshow('image',img)
cv2.imshow('DX',dx)
cv2.imshow('DY',dy)
cv2.imshow('Sobel',sobel_img.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()


# ## Repeat the same with the built-in sobel operator

# In[7]:


sobelx = cv2.Sobel(img, -1, 1, 0, ksize=3)
sobely = cv2.Sobel(img, -1, 0, 1, ksize=3)

sobel_final = cv2.add(sobelx,sobely)


# ## Results

# In[8]:


cv2.imshow('Sobel X', sobelx)
cv2.imshow('Sobel Y', sobely)
cv2.imshow('Sobel Result', sobel_final)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # Canny Edge Detector

# Already shown in lecture. It consists of a few steps. You can think of it this way, whatever the Sobel Operator gives out, it will take that and do a few extra steps. These extra steps are <b>non maximum suppression</b> (i.e. in the direction of the edge orientation) and <b>thresholding</b>. However, opencv is going to do all of these for you.

# In[11]:


canny_edges = cv2.Canny(img, 30, 230) # Play around with the values

cv2.imshow('Canny Edges', canny_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # Gaussian Pyramid

# The gaussian pyramid of various resolution can be generated in Opencv using pyrdown. Pyrdown stands for pyramid down. It gives you a smaller sized image then the original one.

# In[12]:


# generate Gaussian pyramid
image_clone = img.copy() 
g_pyramid = [image_clone] 

for i in range(3):
    G = cv2.pyrDown(g_pyramid[i]) # notice indexing
    cv2.imshow("Down Image " + str(i+1), G)
    g_pyramid.append(G)
    cv2.waitKey(0)
cv2.destroyAllWindows()

#len(g_pyramid)


# Upsampling the downsampled images

# In[13]:


upsampled = g_pyramid[-1]
for i in range(3):
    upsampled = cv2.pyrUp(upsampled)
    cv2.imshow("Up Image " + str(i), upsampled)
    cv2.waitKey(0)
cv2.destroyAllWindows()


# We can generate a laplacian pyramid from the upsampled and downsampled images

# In[14]:


# generate Laplacian Pyramid for A
l_pyramid = [g_pyramid[3]]
for i in range(3,0,-1):
    GE = cv2.pyrUp(g_pyramid[i], dstsize = (g_pyramid[i-1].shape[1], g_pyramid[i-1].shape[0]))
    L = cv2.subtract(g_pyramid[i-1], GE)
    cv2.imshow("L Image " + str(i), L)
    cv2.waitKey(0)
    l_pyramid.append(L)
cv2.destroyAllWindows()


# Now using the details from the laplacian pyramid we try to reconstruct the original images from the downsampled images. A lot of details will come back in this way.

# In[15]:


# now reconstruct
ls = l_pyramid[0]
for i in range(1,4):
    ls = cv2.pyrUp(ls, dstsize = (l_pyramid[i].shape[1], l_pyramid[i].shape[0]))
    ls = cv2.add(ls, l_pyramid[i])
    cv2.imshow("Up Laplacian Image " + str(i), ls)
    cv2.waitKey(0)
cv2.destroyAllWindows()


# # Resources to check out

# References
# 
# <i>https://stackoverflow.com/questions/45817037/opencv-image-subtraction-vs-numpy-subtraction</i>

# # Blending -- Optional for now

# 
# -- Load two images
# 
# -- Create separate gaussian pyramids for both of the images
# 
# -- Create laplacian pyramids for the images as well
# 
# -- Join both of the images on each level of the laplacian pyramids (one half should come from one image, and the other half from the other)
# 
# -- Finally using this newly created laplacian pyramid create the reconstructed blended image
# 
# 

# In[16]:


apple = cv2.imread('imgs/apple.jpg')
orange = cv2.imread('imgs/orange.jpg')
print(apple.shape)
print(orange.shape)
apple_orange = np.hstack((apple[:, :154], orange[:, 154:]))
cv2.imshow("Apple orange", apple_orange)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[17]:


# generate Gaussian pyramid for apple
apple_copy = apple.copy()
gp_apple = [apple_copy]
for i in range(6):
    apple_copy = cv2.pyrDown(apple_copy)
    gp_apple.append(apple_copy)


# generate Gaussian pyramid for orange
orange_copy = orange.copy()
gp_orange = [orange_copy]
for i in range(6):
    orange_copy = cv2.pyrDown(orange_copy)
    gp_orange.append(orange_copy)


# In[18]:


# generate Laplacian Pyramid for apple
apple_copy = gp_apple[5]
lp_apple = [apple_copy]
for i in range(5, 0, -1):
    gaussian_expanded = cv2.pyrUp(gp_apple[i], dstsize = (gp_apple[i-1].shape[1], gp_apple[i-1].shape[0]))
    laplacian = cv2.subtract(gp_apple[i-1], gaussian_expanded)
    lp_apple.append(laplacian)

# generate Laplacian Pyramid for orange
orange_copy = gp_orange[5]
lp_orange = [orange_copy]
for i in range(5, 0, -1):
    gaussian_expanded = cv2.pyrUp(gp_orange[i],  dstsize = (gp_orange[i-1].shape[1], gp_orange[i-1].shape[0]))
    laplacian = cv2.subtract(gp_orange[i-1], gaussian_expanded)
    lp_orange.append(laplacian)


# In[19]:


# Now add left and right halves of images in each level
apple_orange_pyramid = []
n = 0
for apple_lap, orange_lap in zip(lp_apple, lp_orange):
    n += 1
    cols, rows, ch = apple_lap.shape
    laplacian = np.hstack((apple_lap[:, 0:int(cols/2)], orange_lap[:, int(cols/2):]))
    apple_orange_pyramid.append(laplacian)
# now reconstruct
apple_orange_reconstruct = apple_orange_pyramid[0]
for i in range(1, 6):
    apple_orange_reconstruct = cv2.pyrUp(apple_orange_reconstruct, dstsize = (apple_orange_pyramid[i].shape[1], 
                                                                             apple_orange_pyramid[i].shape[0]))
    apple_orange_reconstruct = cv2.add(apple_orange_pyramid[i], apple_orange_reconstruct)


# In[20]:


cv2.imshow("apple", apple)
cv2.imshow("orange", orange)
cv2.imshow("apple_orange", apple_orange)
cv2.imshow("apple_orange_reconstruct", apple_orange_reconstruct)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




