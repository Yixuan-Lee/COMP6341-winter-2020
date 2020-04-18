#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

# In[12]:


import cv2
import numpy as np


# ## Read a single RGB image

# In[13]:


path = './'
file = 'receipt.jpg'

image = cv2.imread(path + file)
cv2.imshow("Original", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ## RGB -> Grayscale -> Blur -> Detect edges

# In[14]:


# Convert the image to grayscale, blur it, and find edges

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 40, 200)
 
# Show the original image and the edge detected image

cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ## Edges detected, proceed with contours

# ### What are contours?

# Contours can be explained simply as a curve joining all the continuous points (along the boundary), having same color or intensity. The contours are a useful tool for shape analysis and object detection and recognition.

# In[16]:


# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = sorted(contours, key = cv2.contourArea, reverse = True)[0]
#print(len(largest_contour[0]))


# ### Check the largest contour

# In[17]:


cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 0)
cv2.imshow("Int", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### Get the length to find the largest closed area and approximate its skeletal structure with just a few points.

# In[18]:


peri = cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True).squeeze()

print(peri)


# ### Display final contour with just 4 points and also show overlay

# In[19]:


blank = np.zeros(image.shape)
overlay = image.copy()
cv2.drawContours(overlay, [approx], -1, (0, 255, 0), 2)
cv2.imshow("Outline on Image", overlay)
cv2.drawContours(blank, [approx], -1, (255, 255, 255), 2)
cv2.imshow("Outline", blank)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ### Finally, transform the image 

# In[20]:


img_points = np.array([[0,0],[0, image.shape[0]-1], [image.shape[1]-1, image.shape[0]-1], 
                       [image.shape[1]-1, 0]], np.float32)
print(img_points)


# In[21]:


M = cv2.getPerspectiveTransform(approx.astype(np.float32), img_points)
warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
cv2.imshow("Scanned Image", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(path + 'scanned.jpg', warped)


# In[ ]:




