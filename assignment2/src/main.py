import os
import numbers
import decimal
import cv2 as cv
import numpy as np
from feature_detection import harris_corner_detection
from feature_detection import harris_corner_detection_ref
from feature_description import calc_sift_descriptor_for_all_interest_points
from feature_matching import descriptors_matching


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
            print('invalid path for image 1 (%s) or image 2 (%s)'
                  % (image_1_path, image_2_path))
            valid_files = False

    return image_1, image_2


def read_threshold_harris():
    valid = False
    threshold = 0

    while valid is False:
        try:
            threshold = float(input('Enter the Harris Corner threshold: '))

            if isinstance(threshold, numbers.Number) is True:
                # threshold is number type
                valid = True

        except ValueError:
            print('Message: Harris Corner threshold should be number type!')
            continue

    return threshold


def read_ssd_threshold():
    valid = False
    threshold = 0

    while valid is False:
        try:
            threshold = float(input('Enter the SSD distance threshold: '))

            if isinstance(threshold, numbers.Number) is True:
                valid = True

        except ValueError:
            print('Message: SSD distance threshold should be number type!')
            continue

    return threshold


def read_ratio_test():
    valid = False
    ratio = 0

    while valid is False:
        try:
            ratio = float(input('Enter the ratio test: '))

            if isinstance(ratio, numbers.Number) is True:
                valid = True

        except ValueError:
            print('Message: ratio test should be number type!')
            continue

    return ratio


def main():
    # ################### 1. Get the Input ################### #
    print('-------------- Feature Detection --------------')
    # read the 2 images
    image_1_orig, image_2_orig = read_images()

    # read harris corner detection threshold (recommended: 40 ~ 80)
    threshold_harris = read_threshold_harris()

    # define a list to record the interest points
    interest_points_image_1 = list()
    interest_points_image_2 = list()

    # ################# 2. Feature Detection ################# #
    # execute harris corner detection
    harris_corner_detection(image_1_orig, threshold_harris,
        interest_points_image_1)
    harris_corner_detection(image_2_orig, threshold_harris,
        interest_points_image_2)

    # initialize the placeholder
    harris_out_1 = image_1_orig.copy()
    harris_out_2 = image_2_orig.copy()

    # draw the interest points on the placeholder
    cv.drawKeypoints(
        image=image_1_orig,
        keypoints=interest_points_image_1,
        outImage=harris_out_1,
        color=(255, 0, 0)       # show interest points in blue color
    )
    cv.drawKeypoints(
        image=image_2_orig,
        keypoints=interest_points_image_2,
        outImage=harris_out_2,
        color=(255, 0, 0)  # show interest points in blue color
    )

    # show the output of my harris corner detection
#     cv.imshow('my harris corner output 1', harris_out)
#     cv.imshow('my harris corner output 2', harris_out)

    # show the output of the build-in harris corner detection (just for
    # comparison)
#      harris_out_ref = harris_corner_detection_ref(image_1_orig, 0.01)

    # show the output of build-in harris corner detection function
#     cv.imshow('build-in harris corner', harris_out_ref)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

    print('------------ Feature Detection Done ------------')

    # ################# 3. Feature Description ################# #
    print('-------------- Feature Description --------------')
    # define a dictionary to record the correspondence of interest points and
    # descriptors
    #   key: KeyPoint object
    #   value: descriptor
    interest_points_descriptor_dict_image_1 = dict()
    interest_points_descriptor_dict_image_2 = dict()

    calc_sift_descriptor_for_all_interest_points(image_1_orig,
        interest_points_image_1, interest_points_descriptor_dict_image_1)
    calc_sift_descriptor_for_all_interest_points(image_2_orig,
        interest_points_image_2, interest_points_descriptor_dict_image_2)

    print('------------ Feature Description Done ------------')

    # ################### 4. Feature Matching ################### #
    print('-------------- Feature Matching --------------')
    # read ssd distance threshold (recommended: 500 ~ 800)
    ssd_threshold = read_ssd_threshold()

    # read the ratio test (recommended: 0.75 ~ 0.80)
    ratio_test = read_ratio_test()

    # define a list which stores the matching relationship
    dmatch_list = list()

    # define 2 lists which store the interest points in 2 images
    # correspondingly (index-0 matches to index-0)
    ip_match_image_1 = list()
    ip_match_image_2 = list()

    descriptors_matching(interest_points_descriptor_dict_image_1,
        interest_points_descriptor_dict_image_2, dmatch_list,
        ip_match_image_1, ip_match_image_2, ssd_threshold, ratio_test)

    print('------------ Feature Matching Done ------------')

    # #################### 5. Draw the Match #################### #
    print('---------------- Draw Matching ----------------')
    match_out = cv.drawMatches(img1=image_1_orig, keypoints1=ip_match_image_1,
        img2=image_2_orig, keypoints2=ip_match_image_2,
        matches1to2=dmatch_list, outImg=None)

    print(match_out.shape)

    cv.imshow('Feature Matching', match_out)
    cv.waitKey(0)
    cv.destroyAllWindows()
    print('-------------- Draw Matching Done --------------')


if __name__ == '__main__':
    main()
