import os
import numbers
import decimal
import cv2 as cv
import numpy as np
from feature_detection import harris_corner_detection
from feature_detection import harris_corner_detection_ref
from feature_description import calc_sift_descriptor_for_all_interest_points

image_sets_folder = '../image_sets'


def read_images():
    valid_files = False
    global image_1, image_2
    global image_1_path, image_2_path

    while valid_files is False:
        try:
            # enter 2 images' paths
            image_1_path = input('Enter the image 1 path: ')
            image_2_path = input('Enter the image 2 path: ')

            # get the relative paths
            image_1_path = os.path.join(
                image_sets_folder, image_1_path)
            image_2_path = os.path.join(
                image_sets_folder, image_2_path)

            # check whether 2 image paths are valid
            image_1_valid = os.path.isfile(image_1_path)
            image_2_valid = os.path.isfile(image_2_path)

            if image_1_valid and image_2_valid:
                image_1 = cv.imread(image_1_path)
                image_2 = cv.imread(image_2_path)
                valid_files = True
            else:
                raise IOError
        except IOError:
            print('invalid path for image 1 (%s) or image 2 (%s)' % (image_1_path, image_2_path))
            valid_files = False

    return image_1, image_2


def read_threshold_harris():
    valid = False
    threshold = 0

    while valid is False:
        try:
            threshold = float(input('Enter the Harris Corner Detection threshold: '))

            if isinstance(threshold, numbers.Number) is True:
                # threshold is number type
                valid = True

        except ValueError:
            print('Message: threshold should be number type!')
            continue

    return threshold


def main():
    # ################### 1. Get the Input ################### #
    # read the 2 images
    image_1_orig, image_2_orig = read_images()

    # read harris corner detection threshold (recommended: 40 ~ 50)
    threshold_harris = read_threshold_harris()

    # define a list to record the interest points
    interest_points_image_1 = list()

    # ################# 2. Feature Detection ################# #
    # execute harris corner detection
    harris_corner_detection(image_1_orig, threshold_harris,
        interest_points_image_1)

    # initialize the placeholder
    harris_out = image_1_orig.copy()

    # draw the interest points on the placeholder
    cv.drawKeypoints(
        image=image_1_orig,
        keypoints=interest_points_image_1,
        outImage=harris_out,
        color=(255, 0, 0)       # show interest points in blue color
    )

    # show the output of my harris corner detection
#     cv.imshow('my harris corner output', harris_out)

    # show the output of the build-in harris corner detection (just for
    # comparison)
#      harris_out_ref = harris_corner_detection_ref(image_1_orig, 0.01)

    # show the output of build-in harris corner detection function
#     cv.imshow('build-in harris corner', harris_out_ref)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

    # ################# 3. Feature Description ################# #
    # key: KeyPoint object
    # value: descriptor
    interest_points_descriptor_dict = dict()

    calc_sift_descriptor_for_all_interest_points(image_1_orig,
        interest_points_image_1, interest_points_descriptor_dict)

    print(interest_points_descriptor_dict)

    # ################### 4. Feature Matching ################### #


if __name__ == '__main__':
    main()
