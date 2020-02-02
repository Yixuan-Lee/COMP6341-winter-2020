import cv2 as cv
import numpy as np
from descriptor import sift_descriptor


def is_out_of_bound(width, height, i, j, descriptor):
    """
    decide whether it is out of bound of the image

    :param width:       number of columns in the image
    :param height:      number of rows in the image
    :param i:           x offset of the pixel
    :param j:           y offset of the pixel
    :param descriptor:  descriptor of the interest point
    :return:    return True if out-of-bound
                return False if not
    """
    # x-axis out-of-bound
    if descriptor.x + i < 0:
        return True
    elif descriptor.x + i >= width:
        return True

    # y-axis out-of-bound
    if descriptor.y + j < 0:
        return True
    elif descriptor.y + j >= height:
        return True

    return False


def set_magnitude_and_theta(descriptor, i, j, w_18_18_i, w_18_18_j):
    """
    set the magnitude and the theta at pixel (i, j)

    :param descriptor:      descriptor of the interest point
    :param i:               x index in magnitudes and thetas
    :param y:               y index in magnitudes and thetas
    :param w_18_18_i:       x index in window_18_18
    :param w_18_18_j:       y index in window_18_18
    """
    diff_x = descriptor.window_18_18[w_18_18_i + 1, w_18_18_j] - descriptor.window_18_18[w_18_18_i - 1, w_18_18_j]
    diff_y = descriptor.window_18_18[w_18_18_i, w_18_18_j + 1] - descriptor.window_18_18[w_18_18_i, w_18_18_j - 1]

    # set magnitude
    magnitude = np.sqrt(diff_x ** 2 + diff_y ** 2)
    descriptor.magnitudes[i, j] = magnitude

    # set theta
    theta = (np.arctan2(diff_x, diff_y) + np.pi) * 180 / np.pi
    descriptor.thetas[i, j] = theta


def calc_sift_decriptor(gray_image_orig, descriptor):
    """
    calculate the sift descriptor for each interest point

    :param gray_image_orig: greyscale image
    :param descriptor:      descriptor of an interest point
    """
    # height and width of the grayscale image
    height = gray_image_orig.shape[0]  # number of rows
    width = gray_image_orig.shape[1]   # number of columns

    # Step 1: set the 18x18 window in the descriptor
    half_window_size = 9
    for i in range(-9, 9):
        for j in range(-9, 9):
            if descriptor.interest_point is not None:
                if is_out_of_bound(width, height, i, j, descriptor):
                    continue
                else:
                    p_x = descriptor.x + i        # number of columns in image
                    p_y = descriptor.y + j        # number of rows in image
                    idx_x = i + half_window_size  # make sure positive index
                    idx_y = j + half_window_size  # make sure positive index
                    # copy the pixels from the grey image to 18x18 window in descriptor
                    descriptor.window_18_18[idx_x, idx_y] = float(gray_image_orig[p_y, p_x])

    # Step 2: compute the magnitude and theta for each pixel in 16x16 window
    for i in range(16):
        for j in range(16):
            # set the index in descriptor.window_18_18 due to the 1 padding
            w_18_18_i = i + 1
            w_18_18_j = j + 1

            # set the magnitude at descriptor.magnitudes[i, j] and
            # the theta at descriptor.thetas[i, j]
            set_magnitude_and_theta(descriptor, i, j, w_18_18_i, w_18_18_j)

    # Step 3: compute the orientation histogram for each 4x4 grid cell
    window_size = 16
    grid_cell_idx = 0
    for i in range(0, window_size, 4):
        for j in range(0, window_size, 4):
            # orientation histogram of each cell
            grid_cell_orientations = np.zeros((1, 8), dtype=np.int32)

            for x in range(4):
                for y in range(4):
                    # x and y coordinates in magnitudes and thetas
                    p_x = x + i
                    p_y = y + j

                    # compute the vote for which bin
                    vote = int(descriptor.thetas[p_x, p_y] / 45)
                    if vote == 8:
                        # when descriptor.thetas[p_x, p_y] = 360, vote = 8,
                        # the pixel votes for 7th bin
                        vote = 7

                    # increment the votes in the bin
                    grid_cell_orientations[0, vote] += 1

            # set the orientation histogram of the 4x4 cell to the descriptor
            descriptor.orientation_histogram[grid_cell_idx, :] = grid_cell_orientations
            grid_cell_idx += 1

    # flatten the orientation histogram
    descriptor.flatten_orientation_histogram()

    # Step 4: Threshold normalize the descriptor
    # (however there is always a few elements which are bigger than 0.2)
    descriptor.normalize_descriptor()


def calc_sift_descriptor_for_all_interest_points(image_orig,
        interest_points_list, interest_points_descriptor_dict):
    """
    define a descriptor for all interest points

    :param image_orig:                      original colored image
    :param interest_points_list:            a list of interest points
    :param interest_points_descriptor_dict: dictionary of descriptors
                                                key:    cv.KeyPoint
                                                value:  descriptor
    """
    # convert the colored image to greyscale
    gray_image_orig = cv.cvtColor(image_orig, cv.COLOR_BGR2GRAY)

    for interest_point in interest_points_list:
        # initialize a descriptor
        des = sift_descriptor(interest_point)

        # calculate the descriptor for the interest point
        calc_sift_decriptor(gray_image_orig, des)

        # add the key-value pair to the dictionary
        interest_points_descriptor_dict[interest_point] = des

