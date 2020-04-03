"""
Comment:
Most of the code here are referenced to my assignment 2 implementation in
'feature_detection.py'
"""

import cv2 as cv
import numpy as np


class harris_corner:

    def __init__(self, rgb_image):
        """
        constructor

        :param rgb_image: colored image
        """
        # initialize the images
        self.gray_image = cv.cvtColor(rgb_image, cv.COLOR_BGR2GRAY)
        self.gray_image = np.float32(self.gray_image)

        # initialize the height and width
        self.height = self.gray_image.shape[0]
        self.width = self.gray_image.shape[1]

        # initialize the response matrix R and R_normalized
        self.R = np.zeros((self.height, self.width), dtype=np.float32)
        self.R_normalized = np.zeros((self.height, self.width), dtype=np.float32)

        # initialize the interest point list
        # the list stores interest points after anms or nms
        self.interest_points_list = list()

    def is_local_maxima(self, x, y):
        """
        check whether R[x, y] is the local maxima with in the 3x3 neighborhood

        :param x:       current x position
        :param y:       current y position
        :return: return True if R[x, y] is the maximum in 3x3 neighborhood,
                 return False otherwise
        """
        if self.R[x, y] == 0:
            return False

        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                if i == 0 and j == 0:
                    # skip comparing with itself
                    continue

                compare_x = x + i
                compare_y = y + j

                if compare_x < 0 or compare_y < 0 \
                        or compare_x >= self.height \
                        or compare_y >= self.width:
                    # handle IndexOutOfBound conditions
                    continue

                if self.R[x, y] < self.R[compare_x, compare_y]:
                    # (x, y) is not the local maxima in the 3x3 neighborhood
                    return False
        return True

    def suppress_neighborhood(self, x, y):
        """
        suppress responses of neighborhood of (x, y) to 0

        :param x:               current x position
        :param y:               current y position
        """
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                if i == 0 and j == 0:
                    # don't suppress current position
                    continue

                suppress_x = x + i
                suppress_y = y + j

                if suppress_x < 0 or suppress_y < 0 \
                        or suppress_x >= self.height \
                        or suppress_y >= self.width:
                    # handle IndexOutOfBound conditions
                    continue

                # suppress the neighbors
                self.R_normalized[suppress_x, suppress_y] = 0

    def non_maximum_suppression(self, interest_point_after_thresholding):
        """
        Apply non-maximum suppression on interest_point_after_thresholding

        :param interest_point_after_thresholding: list of interest points before non-maximum suppression
        """
        for ip in interest_point_after_thresholding:
            x = int(ip.pt[0])   # x -> width  (float->int)
            y = int(ip.pt[1])   # y -> height (float->int)

            if self.is_local_maxima(y, x):
                # keep current response and suppress the neighborhood
                self.suppress_neighborhood(y, x)

                # record the interest point' position
                # Attention: here is (y, x) not (x, y)!! because y is on
                # x-axis, and x is on y-axis
                interest_point = cv.KeyPoint(x=x, y=y, _size=5, _angle=-1)
                interest_point.response = self.R_normalized[y, x]  # set response
                self.interest_points_list.append(interest_point)

            else:
                # keep neighborhood and suppress the current pixel's response
                # to 0
                self.R_normalized[y, x] = 0

    def dist(self, x_i, x_j):
        """
        Compute the distance between x_i and x_j
        :param x_i: an interest point
        :param x_j: another interest point
        :return: distance between x_i and x_j
        """
        diff_x = x_i.pt[0] - x_j.pt[0]
        diff_y = x_i.pt[1] - x_j.pt[1]
        return np.sqrt(diff_x * diff_x + diff_y * diff_y)


    def adaptive_suppression_within_r(self, suppression_radius_r,
            interest_point_after_thresholding, c_robust):
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

                if self.dist(ip, suppress_ip) < suppression_radius_r[ip] \
                        and ip.response > c_robust * suppress_ip.response:
                    # suppress suppress_ip
                    suppress_ip.response = 0

        # scan through interest_point_after_thresholding, add all
        # non-zero-response interest points to self,interest_points_list
        for ip in interest_point_after_thresholding:
            if ip.response > 0:
                self.interest_points_list.append(ip)

    def harris_corner_detection(self, threshold):
        """
        My Harris Corner Detection implementation

        :param threshold:                   harris detection threshold
        """
        # Step 1.A: Compute the x and y derivatives on the image
        # Note: there are 2 functions for derivatives computing:
        #   cv.Sobel()
        #   cv.Scharr()
        # difference: cv.Scharr() calculates a more accurate derivative for a
        # kernel of size 3x3
        Ix = cv.Scharr(src=self.gray_image, ddepth=cv.CV_32F, dx=1, dy=0)
        Iy = cv.Scharr(src=self.gray_image, ddepth=cv.CV_32F, dx=0, dy=1)

        # Step 1.B: Compute the covariance matrix H of the image derivatives
        Ix2 = np.multiply(Ix, Ix)
        Iy2 = np.multiply(Iy, Iy)
        IxIy = np.multiply(Ix, Iy)

        # smooth the result for better detection of corners using a Gaussian
        # weighted window
        gaussian_Ix2 = cv.GaussianBlur(src=Ix2, ksize=(5, 5), sigmaX=0,
            sigmaY=0, borderType=cv.BORDER_DEFAULT)
        gaussian_Iy2 = cv.GaussianBlur(src=Iy2, ksize=(5, 5), sigmaX=0,
            sigmaY=0, borderType=cv.BORDER_DEFAULT)
        gaussian_IxIy = cv.GaussianBlur(src=IxIy, ksize=(5, 5), sigmaX=0,
            sigmaY=0, borderType=cv.BORDER_DEFAULT)

        # Step 1.C: Compute Harris Corner Response R for each pixel using
        #           determinant(H) / trace(H)
        # compute corner response function for each pixel
        for i in range(self.height):
            for j in range(self.width):
                a = gaussian_Ix2[i, j]
                b = gaussian_IxIy[i, j]
                c = gaussian_IxIy[i, j]
                d = gaussian_Iy2[i, j]

                det = a * d - b * c     # determinant(H)
                trace = a + d           # trace(H)

                if trace == 0:
                    # avoid zero denominator
                    self.R[i, j] = 0
                else:
                    # c[H] value in guideline
                    self.R[i, j] = det / trace

        # scale the response of the Harris Corner to lie between 0 and 255
        self.R_normalized = cv.normalize(src=self.R, dst=None, alpha=0.0,
            beta=255.0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

        # Step 1.D: Threshold R then Find peaks in the response matrix and
        # store the interest point locations
        # Thresholding
        interest_point_after_thresholding = list()
        for i in range(self.height):
            for j in range(self.width):
                if self.R_normalized[i, j] <= threshold:
                    self.R_normalized[i, j] = 0
                else:
                    # create a KeyPoint object and insert into the list
                    interest_point = cv.KeyPoint(x=j, y=i, _size=5, _angle=-1,
                        _response=self.R_normalized[i, j])
                    interest_point_after_thresholding.append(interest_point)

        # Find peaks in the response matrix and store the interest
        # point locations (use adaptive non-maximum suppression by default)
        anms = True  # adaptive non-maximum suppression identifier
        if anms is True:
            # adaptive non-maximum suppression
            r = 24
            c_robust = 0.9  # robust value

            # ANMS step 1) Find the global maximum
            max_ip_response = 0.0
            max_ip = None
            for ip in interest_point_after_thresholding:
                if ip.response > max_ip_response:
                    max_ip_response = ip.response
                    max_ip = ip

            # ANMS step 2) Append the global maximum ip to interest point list
            # this will cause global maximum interest point added twice
            # self.interest_points_list.append(max_ip)

            # ANMS step 3) Scan through all interest points again and
            #              compute the r_i for each interest point
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
                        # if the interest point's response < robust_response, then
                        # we need to calculate the minimum suppression radius r_i
                        distance = self.dist(x_i, x_j)
                        if distance < min_dist:
                            min_dist = distance

                r_i = min_dist

                suppression_radius_dict[x_i] = r_i

            # ANMS step 4) then apply non-maximum suppression within the r_i
            # for each interest point
            self.adaptive_suppression_within_r(suppression_radius_dict,
                interest_point_after_thresholding, c_robust)
        else:
            # non-maximum suppression within 3x3 neighborhood
            self.non_maximum_suppression(interest_point_after_thresholding)

