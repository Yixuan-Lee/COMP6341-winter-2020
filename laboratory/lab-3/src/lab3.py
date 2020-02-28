"""
Today's topic: Transformation

Office hour: 3-4 pm Thursday
"""

import cv2
import numpy as np

path = '../'
file = 'image.jpeg'

"""
1 0 tx
0 1 ty
"""
# tx, ty represent how much we want to shift the image
# two 1s here control the stretching on the image
# cv2.warpAffine: does the affine transformation


# Rotation
# cv2.getRotationMatrix2D:
#   90: angle
#   (col/2, rows/2): rotate centering point
#   1:


# Scaling
# cv2.resize(fx=how scale on x, fy=how scale on y, interpolation)

# Translation using a set of points (**important)
# giving 3 points, we move an image to a new location
# cv2.getAffineTransform(): gives the transform matrix (there is a better function, google it, perspective_transform)
# cv2.getAffineTransform():

# Looping
# scaling
# 97: small a key
#   press 'a' to rescale the image
# 27: escape key ('ESC')
#   press 'esc' to break the loop

# Rotating
# 115: small 's' key


# Extras
# 1. Harris corner
# 2. blending (in project)

