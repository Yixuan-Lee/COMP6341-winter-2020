"""
Comment:
Most of the code here are referenced to my assignment 2 implementation in
'feature_description.py'
"""

import cv2 as cv
import numpy as np
from sift_descriptor import sift

"""
the class is used for calculating and containing all descriptors for all 
interest points in the image
"""
class descriptors:

    def __init__(self, rgb_image, harris_interest_points_list):
        """
        constructor

        :param rgb_image: original colored image
        :param harris_interest_points_list: harris output of interest_points_list
        """

        # initialize a dictionary of interest_point -> descriptor
        #       key: interest point
        #       value: sift_descriptor of the interest point
        self.interest_point_descriptor_dict = dict()

        # initialize the image
        self.gray_image = cv.cvtColor(rgb_image, cv.COLOR_BGR2GRAY)
        self.gray_image = np.float32(self.gray_image)

        # initialize the height and width
        self.height = self.gray_image.shape[0]  # number of rows
        self.width = self.gray_image.shape[1]   # number of columns

        # set the interest points list
        self.interest_points_list = harris_interest_points_list

    def is_out_of_bound(self, i, j, descriptor):
        """
        decide whether it is out of bound of the image

        :param i:           x offset of the pixel
        :param j:           y offset of the pixel
        :param descriptor:  descriptor of the interest point
        :return:    return True if out-of-bound
                    return False if not
        """
        # x-axis out-of-bound
        if descriptor.x + i < 0:
            return True
        elif descriptor.x + i >= self.width:
            return True

        # y-axis out-of-bound
        if descriptor.y + j < 0:
            return True
        elif descriptor.y + j >= self.height:
            return True

        return False

    def calc_sift_decriptor(self, descriptor):
        """
        calculate the sift descriptor for each interest point

        :param gray_image_orig: greyscale image
        :param descriptor:      descriptor of an interest point
        """
        # set the 18x18 window in the descriptor
        half_window_size = 9
        for i in range(-9, 9):
            for j in range(-9, 9):
                if descriptor.interest_point is not None:
                    if self.is_out_of_bound(i, j, descriptor):
                        continue
                    else:
                        # set up indices used to access in image and in window
                        p_x = descriptor.x + i    # number of columns in image
                        p_y = descriptor.y + j    # number of rows in image
                        idx_x = i + half_window_size  # ensure positive index
                        idx_y = j + half_window_size  # ensure positive index

                        # copy the pixels from the grey image to 18x18 window
                        # in descriptor
                        descriptor.window_18_18[idx_x, idx_y] = \
                            float(self.gray_image[p_y, p_x])

        # compute the magnitude and theta for each pixel in 16x16 window
        descriptor.set_magnitude_and_theta()

        # compute the orientation histogram for each 4x4 grid cell
        descriptor.set_orientation_histogram_for_grid_cells()

        # threshold and normalize the descriptor
        descriptor.normalize_descriptor()

    def calc_sift_descriptor_for_all_interest_points(self):
        """
        define a descriptor for all interest points
        """
        for interest_point in self.interest_points_list:
            # initialize a descriptor
            descriptor = sift(interest_point)

            # calculate the descriptor for the interest point
            self.calc_sift_decriptor(descriptor)

            # add the key-value pair to the dictionary
            self.interest_point_descriptor_dict[interest_point] = descriptor

