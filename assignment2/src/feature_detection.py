import cv2 as cv
import numpy as np

# debugging indicators
trace_dtype = False
trace_shape = False


def is_local_maxima(R, x, y, height, width):
    """
    check whether R[x, y] is the local maxima with in the 3x3 neighborhood

    :param R:       response matrix
    :param x:       current x position
    :param y:       current y position
    :param height:  height of R
    :param width:   width of R
    :return: return True if R[x, y] is the maximum in 3x3 neighborhood,
             return False otherwise
    """
    if R[x, y] == 0:
        return False

    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            if i == 0 and j == 0:
                # skip comparing with itself
                continue

            compare_x = x + i
            compare_y = y + j

            if compare_x < 0 or compare_y < 0 \
                    or compare_x >= height or compare_y >= width:
                # handle IndexOutOfBound conditions
                continue

            if R[x, y] < R[compare_x, compare_y]:
                # (x, y) is not the local maxima in the 3x3 neighborhood
                return False
    return True


def suppress_neighborhood(R, x, y, height, width):
    """
    suppress responses of neighborhood of (x, y)

    :param R:       response matrix
    :param x:       current x position
    :param y:       current y position
    :param height:  height of R
    :param width:   width of R
    """
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            if i == 0 and j == 0:
                # don't suppress current position
                continue

            suppress_x = x + i
            suppress_y = y + j

            if suppress_x < 0 or suppress_y < 0 \
                    or suppress_x >= height or suppress_y >= width:
                # handle IndexOutOfBound conditions
                continue

            R[suppress_x, suppress_y] = 0


def harris_corner_detection_ref(image_orig, threshold):
    """
    Harris Corner Detector using OpenCV library function directly. Show the
    interest points on the colored image
    (Used for comparison with my Harris Corner Implementation)

    Reference:
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html#harris-corner-detector-in-opencv

    :param image_orig:  original colored image
    :param threshold:   harris detection threshold
    """
    # convert the colored image to greyscale and transfer to type float32
    gray_image_orig = cv.cvtColor(image_orig, cv.COLOR_BGR2GRAY)
    gray_image_orig = np.float32(gray_image_orig)

    # apply harris corner
    dst = cv.cornerHarris(src=gray_image_orig, blockSize=2, ksize=3, k=0.04)
    # result is dilated for marking the corners
    dst = cv.dilate(dst, None)

    # thresholding and marking
    image_orig[dst > threshold * dst.max()] = [0, 0, 255]

    # show the output of build-in harris corner detection function
    cv.imshow('build-in harris corner', image_orig)


def harris_corner_detection(image_orig, threshold, interest_points_list):
    """
    My Harris Corner Detection implementation

    :param image_orig:              original colored image
    :param threshold:               harris detection threshold
    :param interest_points_list     records the positions of interest points
    """
    # convert the colored image to greyscale
    gray_image_orig = cv.cvtColor(image_orig, cv.COLOR_BGR2GRAY)
    # transfer to type float32
    gray_image_orig = np.float32(gray_image_orig)

    # Step 1: Compute derivatives Ix^2, Iy^2 and IxIy at each pixel and
    #         smooth them with a 5x5 Gaussian
    # compute the x-axis derivative Ix and y-axis derivative Iy and
    # cross multiplication Ixy
    Ix = cv.Scharr(src=gray_image_orig, ddepth=cv.CV_32F, dx=1, dy=0)
    Iy = cv.Scharr(src=gray_image_orig, ddepth=cv.CV_32F, dx=0, dy=1)

    # show the gradients images
    # gradient_x = cv.normalize(src=Ix, dst=None, alpha=0.0, beta=1.0,
    #     norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    # gradient_y = cv.normalize(src=Iy, dst=None, alpha=0.0, beta=1.0,
    #     norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    # cv.imshow('gradient x', gradient_x)
    # cv.imshow('gradient y', gradient_y)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # Step 2: Compute the 3 elements in Harris Matrix H (formula in guideline)
    Ix2 = np.multiply(Ix, Ix)
    Iy2 = np.multiply(Iy, Iy)
    IxIy = np.multiply(Ix, Iy)

    # smooth Ix, Iy, IxIy with a 5x5 Gaussian
    gaussian_Ix2 = cv.GaussianBlur(src=Ix2, ksize=(5, 5), sigmaX=0, sigmaY=0,
        borderType=cv.BORDER_DEFAULT)
    gaussian_Iy2 = cv.GaussianBlur(src=Iy2, ksize=(5, 5), sigmaX=0, sigmaY=0,
        borderType=cv.BORDER_DEFAULT)
    gaussian_IxIy = cv.GaussianBlur(src=IxIy, ksize=(5, 5), sigmaX=0,
        sigmaY=0, borderType=cv.BORDER_DEFAULT)

    if trace_dtype is True:
        print('gaussian_Ix2.dtype =', gaussian_Ix2.dtype)   # float32
        print('gaussian_Iy2.dtype =', gaussian_Iy2.dtype)   # float32
        print('gaussian_IxIy.dtype =', gaussian_IxIy.dtype) # float32

    if trace_shape is True:
        print('gray_image_orig.shape', gray_image_orig.shape)
        print('gaussian_Ix2.shape =', gaussian_Ix2.shape)  # same as gray_image_orig.shape
        print('gaussian_Iy2.shape =', gaussian_Iy2.shape)  # same as gray_image_orig.shape
        print('gaussian_IxIy.shape =', gaussian_IxIy.shape)# same as gray_image_orig.shape

    # Step 3: Compute corner response function R for each pixel
    # initialize the response matrix R
    height = gray_image_orig.shape[0]
    width = gray_image_orig.shape[1]
    R = np.zeros((height, width), dtype=np.float32)
    alpha = 0.04    # recommend value 0.04 ~ 0.06

    # compute corner response function for each pixel
    for i in range(height):
        for j in range(width):
            a = gaussian_Ix2[i, j]
            b = gaussian_IxIy[i, j]
            c = gaussian_IxIy[i, j]
            d = gaussian_Iy2[i, j]

            det = a * d - b * c     # determinant
            trace = a + d           # trace

            if trace == 0:
                # avoid zero denominator
                R[i, j] = 0
            else:
                # c[H] value in guideline
                R[i, j] = det / trace

    # Extra step: normalize R to 0 ~ 255 because value in R are so huge
    # leading to large variation
    R_normalized = cv.normalize(src=R, dst=None, alpha=0.0, beta=255.0,
        norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

    # Step 4: Threshold R
    for i in range(height):
        for j in range(width):
            if R_normalized[i, j] <= threshold:
                R_normalized[i, j] = 0

    # Step 5: Find local maxima of response function for each pixel in 3x3
    #         neighborhood (a.k.a non-maximum suppression)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if is_local_maxima(R_normalized, i, j, height, width):
                # keep current response and suppress the neighborhood
                suppress_neighborhood(R_normalized, i, j, height, width)

                # record the interest point' position
                # Attention: here is (j, i) not (i, j)!! because j is on
                # x-axis, and i is on y-axis
                interest_point = cv.KeyPoint(j, i, 5)
                interest_point.response = R_normalized[i, j]  # set response
                interest_points_list.append(interest_point)

                # print the interest point's position
                # print('(%d, %d)' % (i, j))
            else:
                # keep neighborhood and suppress current response
                R_normalized[i, j] = 0

