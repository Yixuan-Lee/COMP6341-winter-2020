import cv2 as cv
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
                    # set up indices used to access in image and in window
                    p_x = descriptor.x + i        # number of columns in image
                    p_y = descriptor.y + j        # number of rows in image
                    idx_x = i + half_window_size  # make sure positive index
                    idx_y = j + half_window_size  # make sure positive index

                    # copy the pixels from the grey image to 18x18 window
                    # in descriptor
                    descriptor.window_18_18[idx_x, idx_y] = float(gray_image_orig[p_y, p_x])

    # Step 2: compute the magnitude and theta for each pixel in 16x16 window
    descriptor.set_magnitude_and_theta()

    # Step 3: compute the orientation histogram for each 4x4 grid cell
    descriptor.set_orientation_histogram_for_grid_cells()

    # Step 4: Threshold normalize the descriptor
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

