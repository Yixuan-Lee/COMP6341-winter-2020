#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import cv2
import numpy as np


# #### File paths

# In[2]:


path = '../'
file = 'image.jpeg'


# # Loading images

# In[8]:


img = cv2.imread(path + file)
rows,cols, channels = img.shape
rows, cols, channels


# # Translation

# M is basically the transformation matrix. In this case the one I am using is a 2 by 3 transformation matrix. Almost like this,
# 
# 1 0 tx
# 
# 0 1 ty

# In[ ]:


M = np.float32([[1,0,100],
                [0,1,100]])
dst = cv2.warpAffine(img,M,(cols,rows))

cv2.imshow('img',img)
cv2.imshow('Translated',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # Rotation

# In[3]:


img = cv2.imread(path + file)
rows,cols,channels = img.shape

M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
print(M)


# In[4]:


dst = cv2.warpAffine(img,M,(cols,rows))

print(M)

cv2.imshow('img',img)
cv2.imshow('Rotated',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # Scaling

# Scaling changes the dimension of your image. You take an image in, you define the scale factor along each axis and out comes a new image with new dimensions. These new dimensions are calculated using the scale factors and the original image resolution.

# In[5]:


img = cv2.imread(path + file)
res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)

cv2.imshow('img',img)
cv2.imshow('Scaled Image',res)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # Translation using a set of points

# Translation moves your image to a particular direction based on whichever direction you want it to move.

# In[3]:


import numpy as np
import cv2

img = cv2.imread('2.jpg')
rows,cols,ch = img.shape

pts1 = np.float32([[0,0],[0, 1],[1, 0]])
pts2 = np.float32([[0.6,0.2],[0.1,0.3],[1,0.3]])

M = cv2.getAffineTransform(pts1,pts2)
print(M)

dst = cv2.warpAffine(img,M,(cols,rows))
cv2.imshow('Translated',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # Looping

# ### Scaling

# In[4]:


img = cv2.imread(path + file)
cv2.imshow('Scaled image', img)
resize_value = 1

while(True):
    key = cv2.waitKey(0)
    if (key == 27):
        cv2.destroyAllWindows()
        break
    elif (key == 97):
        resize_value += .5
        res = cv2.resize(img,None,fx=resize_value, fy=resize_value, interpolation = cv2.INTER_CUBIC)

        cv2.imshow('Scaled Image',res) 


# ### Rotating

# In[ ]:


img = cv2.imread(path + file)
cv2.imshow('Scaled image', img)
rows,cols,channels = img.shape
rotation_value = 0

while(True):
    key = cv2.waitKey(0)
    if (key == 27):
        cv2.destroyAllWindows()
        break
    elif (key == 97):
        rotation_value += 15
        M = cv2.getRotationMatrix2D((cols/2,rows/2),rotation_value,1)
        res = cv2.warpAffine(img,M,(cols,rows))
        cv2.imshow('Scaled Image',res) 
    elif (key == 115):
        rotation_value -= 15
        M = cv2.getRotationMatrix2D((cols/2,rows/2),rotation_value,1)
        res = cv2.warpAffine(img,M,(cols,rows))
        cv2.imshow('Scaled Image',res) 


# # Extras (Not to be covered, probably)

# In[3]:


import cv2
import numpy as np

source_window = 'Source image'
corners_window = 'Corners detected'
max_thresh = 255

def cornerHarris_demo(val):
    thresh = val
    # Detector parameters
    blockSize = 2
    apertureSize = 3
    k = 0.04
    # Detecting corners
    dst = cv2.cornerHarris(src_gray, blockSize, apertureSize, k)
    # Normalizing
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    dst_norm_scaled = cv2.convertScaleAbs(dst_norm)
    # Drawing a circle around corners
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i,j]) > thresh:
                cv2.circle(dst_norm_scaled, (j,i), 5, (0), 2)
    # Showing the result
    cv2.namedWindow(corners_window)
    cv2.imshow(corners_window, dst_norm_scaled)

src = cv2.imread('./boxes.jpg')
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# Create a window and a trackbar
cv2.namedWindow(source_window)
thresh = 200 # initial threshold

cv2.createTrackbar('Threshold: ', source_window, thresh, max_thresh, cornerHarris_demo)
cv2.imshow(source_window, src)
cornerHarris_demo(thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # More blending

# In[4]:


import cv2
import numpy as np

tower = cv2.imread('1.jpg')
sea = cv2.imread('2.jpg')
print(tower.shape)
print(sea.shape)
tower_sea = np.vstack((tower[:500, :], sea[500:, :]))
cv2.imshow("Tower Sea", tower_sea)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[5]:


# generate Gaussian pyramid for apple
tower_copy = tower.copy()
gp_tower = [tower_copy]
for i in range(6):
    tower_copy = cv2.pyrDown(tower_copy)
    gp_tower.append(tower_copy)


# generate Gaussian pyramid for orange
sea_copy = sea.copy()
gp_sea = [sea_copy]
for i in range(6):
    sea_copy = cv2.pyrDown(sea_copy)
    gp_sea.append(sea_copy)


# In[6]:


# generate Laplacian Pyramid for apple
tower_copy = gp_tower[5]
lp_tower = [tower_copy]
for i in range(5, 0, -1):
    gaussian_expanded = cv2.pyrUp(gp_tower[i], dstsize = (gp_tower[i-1].shape[1], gp_tower[i-1].shape[0]))
    laplacian = cv2.subtract(gp_tower[i-1], gaussian_expanded)
    lp_tower.append(laplacian)

# generate Laplacian Pyramid for orange
sea_copy = gp_sea[5]
lp_sea = [sea_copy]
for i in range(5, 0, -1):
    gaussian_expanded = cv2.pyrUp(gp_sea[i],  dstsize = (gp_sea[i-1].shape[1], gp_sea[i-1].shape[0]))
    laplacian = cv2.subtract(gp_sea[i-1], gaussian_expanded)
    lp_sea.append(laplacian)


# In[7]:


# Now add left and right halves of images in each level
tower_sea_pyramid = []
n = 0
for tower_lap, sea_lap in zip(lp_tower, lp_sea):
    n += 1
    cols, rows, ch = tower_lap.shape
    laplacian = np.vstack((tower_lap[0:int((rows/2)), :], sea_lap[int(rows/2):, :]))
    tower_sea_pyramid.append(laplacian)
# now reconstruct
tower_sea_reconstruct = tower_sea_pyramid[0]
for i in range(1, 6):
    tower_sea_reconstruct = cv2.pyrUp(tower_sea_reconstruct, dstsize = (tower_sea_pyramid[i].shape[1], 
                                                                             tower_sea_pyramid[i].shape[0]))
    tower_sea_reconstruct = cv2.add(tower_sea_pyramid[i], tower_sea_reconstruct)


# In[8]:


cv2.imshow("Eiffel Tower", tower)
cv2.imshow("Sea", sea)
cv2.imshow("Tower_Sea", tower_sea)
cv2.imshow("Tower_Sea Reconstructed", tower_sea_reconstruct)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




