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

            if compare_x < 0 or compare_y < 0 or compare_x >= height or compare_y >= width:
                # handle IndexOutOfBound conditions
                continue

            if R[x, y] < R[compare_x, compare_y]:
                # (x, y) is not the local maxima in the 3x3 neighborhood
                return False
    return True


def suppress_neighborhood(R_normalized, x, y, height, width):
    """
    suppress responses of neighborhood of (x, y) to 0

    :param R_normalized:    response matrix
    :param x:               current x position
    :param y:               current y position
    :param height:          height of R
    :param width:           width of R
    """
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            if i == 0 and j == 0:
                # don't suppress current position
                continue

            suppress_x = x + i
            suppress_y = y + j

            if suppress_x < 0 or suppress_y < 0 or suppress_x >= height or suppress_y >= width:
                # handle IndexOutOfBound conditions
                continue

            # suppress the neighbors
            R_normalized[suppress_x, suppress_y] = 0


def non_maximum_suppression(R_normalized, height, width,
        interest_point_before_suppression, interest_point_after_suppression):
    """
    Apply non-maximum suppression on interest_point_after_thresholding

    :param R_normalized:
    :param height:
    :param width:
    :param interest_point_before_suppression:
    :param interest_point_after_suppression:
    """
    for ip in interest_point_before_suppression:
        x = int(ip.pt[0])   # x -> width
        y = int(ip.pt[1])   # y -> height

        if is_local_maxima(R_normalized, y, x, height, width):
            # keep current response and suppress the neighborhood
            suppress_neighborhood(R_normalized, y, x, height, width)

            # record the interest point' position
            # Attention: here is (j, i) not (i, j)!! because j is on
            # x-axis, and i is on y-axis
            interest_point = cv.KeyPoint(x=x, y=y, _size=5, _angle=-1)
            interest_point.response = R_normalized[y, x]  # set response
            interest_point_after_suppression.append(interest_point)

            # print the interest point's position
            # print('(%d, %d)' % (j, i))
        else:
            # keep neighborhood and suppress the current pixel's response
            # to 0
            R_normalized[y, x] = 0


def dist(x_i, x_j):
    """
    Compute the distance between x_i and x_j
    :param x_i: an interest point
    :param x_j: another interest point
    :return: distance between x_i and x_j
    """
    diff_x = x_i.pt[0] - x_j.pt[0]
    diff_y = x_i.pt[1] - x_j.pt[1]
    return np.sqrt(diff_x * diff_x + diff_y * diff_y)


def adaptive_suppression_within_r(suppression_radius_r,
        anms_interest_points_list, interest_point_after_thresholding,
        c_robust):
    """
    Suppress neighbor interest points within suppression radius

    :param suppression_radius_r:                suppression radius
    :param anms_interest_points_list:           interest points storing the interest point after adaptive suppression
    :param interest_point_after_thresholding:   a list of interest points after thresholding on response
    :param c_robust:                            c_robust parameter discussed in MOPS paper
    """
    for ip in interest_point_after_thresholding:

        for suppress_ip in interest_point_after_thresholding:
            if ip == suppress_ip:
                continue

            if dist(ip, suppress_ip) < suppression_radius_r[ip] and ip.response > c_robust * suppress_ip.response:
                # suppress suppress_ip
                suppress_ip.response = 0

    # scan through interest_point_after_thresholding, add all
    # non-zero-response interest points to anms_interest_points_list
    for ip in interest_point_after_thresholding:
        if ip.response > 0:
            anms_interest_points_list.append(ip)


def harris_corner_detection_ref(image_orig, threshold):
    """
    Harris Corner Detector using OpenCV library function directly. Show the
    interest points on the colored image
    (Only used for comparison with my Harris Corner Implementation)

    Reference:
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html#harris-corner-detector-in-opencv

    :param image_orig:  original colored image
    :param threshold:   harris detection threshold
    :return build-in harris corner output
    """
    # convert the colored image to greyscale and transfer to type float32
    gray_image_orig = cv.cvtColor(image_orig, cv.COLOR_BGR2GRAY)
    gray_image_orig = np.float32(gray_image_orig)

    # apply harris corner
    dst = cv.cornerHarris(src=gray_image_orig, blockSize=2, ksize=3, k=0.04)
    # result is dilated for marking the corners
    dst = cv.dilate(dst, None)

    # thresholding and marking
    harris_out_ref = image_orig.copy()
    harris_out_ref[dst > threshold * dst.max()] = [0, 0, 255]

    return harris_out_ref


def harris_corner_detection(image_orig, threshold, interest_points_list):
    """
    My Harris Corner Detection implementation

    :param image_orig:                  original colored image
    :param threshold:                   harris detection threshold
    :param interest_points_list         records the positions of interest points
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
#     gradient_x = cv.normalize(src=Ix, dst=None, alpha=0.0, beta=1.0,
#         norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
#     gradient_y = cv.normalize(src=Iy, dst=None, alpha=0.0, beta=1.0,
#         norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
#     cv.imshow('gradient x', gradient_x)
#     cv.imshow('gradient y', gradient_y)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

    # Step 2: Compute the 3 elements in Harris Matrix H (formula in guideline)
    ############## 1.1 computation of Harris matrix ##############
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
        print('gaussian_Ix2.shape =', gaussian_Ix2.shape)   # same as gray_image_orig.shape
        print('gaussian_Iy2.shape =', gaussian_Iy2.shape)   # same as gray_image_orig.shape
        print('gaussian_IxIy.shape =', gaussian_IxIy.shape) # same as gray_image_orig.shape

    # Step 3: Compute corner response function R for each pixel
    # initialize the height and width and the response matrix R
    ############## 1.2 computation of response value ##############
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

    # Extra step: normalize R to 0 ~ 255 because values in R may be huge
    # leading to large variation
    R_normalized = cv.normalize(src=R, dst=None, alpha=0.0, beta=255.0,
        norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

    # Step 4: Threshold R
    ############## 1.3 identify interest points ##############
    interest_point_after_thresholding = list()
    for i in range(height):
        for j in range(width):
            if R_normalized[i, j] <= threshold:
                R_normalized[i, j] = 0
            else:
                # create a KeyPoint object and insert into the list
                interest_point = cv.KeyPoint(x=j, y=i, _size=5, _angle=-1,
                    _response=R_normalized[i, j])
                interest_point_after_thresholding.append(interest_point)

    # Step 5: Find local maxima of response function for each pixel in 3x3
    #         neighborhood (a.k.a adaptive non-maximum suppression)
    anms = True  # anms identifier
    if anms is True:
        ############## 1.5 adaptive non-maximum suppression ##############
        r = 24
        c_robust = 0.9  # robust value

        # ANMS step 5.1) Find the global maximum
        max_ip_response = 0.0
        max_ip = None
        for ip in interest_point_after_thresholding:
            if ip.response > max_ip_response:
                max_ip_response = ip.response
                max_ip = ip

        # ANMS step 5.2) Append the global maximum ip to the anms list
        interest_points_list.append(max_ip)

        # ANMS step 5.3) Scan through all interest points again and compute the
        #                r_i for each interest points
        #   key:    interest point
        #   value:  suppression radius
        suppression_radius_dict = dict()
        for x_i in interest_point_after_thresholding:
            f_x_i = x_i.response
            min_dist = r

            for x_j in interest_point_after_thresholding:
                if x_j == x_i:
                    continue

                f_x_j = x_j.response

                if f_x_i < c_robust * f_x_j:
                    # if the interest point's response < robust_response,
                    # then we need to calculate the minimum suppression radius
                    # r_i
                    distance = dist(x_i, x_j)
                    if distance < min_dist:
                        min_dist = distance

            r_i = min_dist

            suppression_radius_dict[x_i] = r_i

        # ANMS step 5.4) then apply non-maximum suppression within the r_i for
        # each interest point
        adaptive_suppression_within_r(suppression_radius_dict,
            interest_points_list, interest_point_after_thresholding,
            c_robust)
    else:
        ############## 1.4 non-maximum suppression ##############
        # apply normal non-maximum suppression if we disable the
        # adaptive suppression
        non_maximum_suppression(R_normalized, height, width,
            interest_point_after_thresholding, interest_points_list)

