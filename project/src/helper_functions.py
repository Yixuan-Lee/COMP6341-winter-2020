import os
import cv2 as cv
import numbers

image_sets_folder = '../project_images'
image_result_folder = '../result_images'


def read_threshold_harris():
    """
    read harris corner detection threshold
    :return: harris corner detection threshold
    """
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
    """
    read SSD distance threshold
    :return: SSD distance threshold
    """
    valid = False
    ssd_threshold = 0

    while valid is False:
        try:
            ssd_threshold = float(input('Enter the SSD distance threshold: '))

            if isinstance(ssd_threshold, numbers.Number) is True:
                valid = True

        except ValueError:
            print('Message: SSD distance threshold should be number type!')
            continue

    return ssd_threshold


def read_ratio_test_threshold():
    """
    read ratio test (best_ssd_distance / second_best_ssd_distance)
    :return: ratio (best_ssd_distance / second_best_ssd_distance)
    """
    valid = False
    ratio = 0

    while valid is False:
        try:
            ratio = float(input('Enter the ratio test threshold: '))

            if isinstance(ratio, numbers.Number) is True:
                valid = True

        except ValueError:
            print('Message: ratio test should be number type!')
            continue

    return ratio


def read_inlier_threshold():
    """
    read inlier threshold
    :return: inlier threshold
    """
    valid = False
    inlier_threshold = 0

    while valid is False:
        try:
            inlier_threshold = float(input('Enter the inlier threshold: '))

            if isinstance(inlier_threshold, numbers.Number) is True:
                valid = True

        except ValueError:
            print('Message: inlier threshold should be number type!')
            continue

    return inlier_threshold


def read_no_of_iterations():
    """
    read number of iterations
    :return: number of iteration of running RANSAC
    """
    valid = False
    num_of_iterations = 0

    while valid is False:
        try:
            num_of_iterations = int(input('Enter number of iterations of RANSAC: '))

            if isinstance(num_of_iterations, numbers.Number) is True:
                valid = True

        except ValueError:
            print('Message: inlier threshold should be number type!')
            continue

    return num_of_iterations


def print_params_table(image_list, harris_threshold, ssd_threshold,
        ratio_test_threshold, inlier_threshold, number_of_iterations):
    """
    print a table of parameters used in the program

    :param image_list:
    :param harris_threshold:
    :param ssd_threshold:
    :param ratio_test_threshold:
    :param inlier_threshold:
    :param number_of_iterations:
    :return:
    """
    print('----------------------- Parameters Table -----------------------')
    print('image_list = ', image_list.img_path_list)
    print('Parameter used in Harris Corner: ')
    print('\t harris corner threshold =', harris_threshold)
    print('Parameters used in Feature Matching: ')
    print('\t ssd distance            =', ssd_threshold)
    print('\t ratio test threshold    =', ratio_test_threshold)
    print('Parameters used in RANSAC: ')
    print('\t inlier threshold        =', inlier_threshold)
    print('\t number of iterations    =', number_of_iterations)
    print('-----------------------------------------------------------------')


def save_image(save_file_name, save_image):
    """
    save the image as "save_file_name.png" under the folder ../result_images

    :param save_file_name: file name to save
    """
    save_path = os.path.join(image_result_folder, save_file_name)
    cv.imwrite(filename=save_path, img=save_image)


