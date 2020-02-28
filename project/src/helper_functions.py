import os
import cv2 as cv
import numbers

image_sets_folder = '../project_images'
image_result_folder = '../result_images'


def read_all_images():
    """
    load all images that are going to be stitched
    (# is the stopping sign)

    :return: a list of images
    """

    print('---- Enter all images\' paths (enter # as image path to end) ----')

    image_list = list()
    global image_path
    image_path = ''
    image_counter = 1

    # read until hitting the stopping sign
    while image_path != '#':

        valid_path = False
        # re-enter the image path if the path is invalid
        while valid_path is False:
            try:
                # enter the image path
                image_path = input('Enter the image path %d: '
                                   % (image_counter))

                # check if the input is stopping sign (#)
                if image_path == '#':
                    break

                # concatenate the relative path
                image_path = os.path.join(image_sets_folder, image_path)

                # check if the image path is valid
                image_path_check = os.path.isfile(image_path)

                if image_path_check is True:
                    image_list.append(cv.imread(image_path))
                    valid_path = True
                    image_counter += 1
                else:
                    raise IOError
            except IOError:
                print('invalid image path for image (%s)' % (image_path))

    return image_list


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


def read_ratio_test_threshold():
    """
    read ratio test (best_ssd_distance / second_best_ssd_distance)
    :return: ratio (best_ssd_distance / second_best_ssd_distance)
    """
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


def save_image(save_file_name, save_image):
    """
    save the image as "save_file_name.png" under the folder ../result_images

    :param save_file_name: file name to save
    """
    save_path = os.path.join(image_result_folder, save_file_name)
    cv.imwrite(filename=save_path, img=save_image)


