import os
import cv2 as cv


image_sets_folder = '../project_images'
image_result_folder = '../result_images'

"""
this class is created for print_params_table(.) function in 
helper_functions.py, because after append cv.imread(image_path) object here, 
I didn't find a way to present the object's image filename. So I create this
class for the convenience of presenting the image filename in 
print_params_table(.) function
"""
class my_image_list:
    def __init__(self):
        self.img_list = list()
        self.img_path_list = list()

    def length(self):
        return len(self.img_list)

    def read_all_images(self):
        """
        load all images that are going to be stitched
        (# is the stopping sign)

        :return: a list of images
        """
        print('---- Enter all images\' paths (enter # as image path to end) ----')
        image_path = ''
        image_counter = 1

        # read until hitting the stopping sign
        while image_path != '#':

            valid_path = False
            # re-enter the image path if the path is invalid
            while valid_path is False:
                try:
                    # enter the image path
                    image_path = input('Enter the image path %d: ' % (image_counter))

                    # check if the input is stopping sign (#)
                    if image_path == '#':
                        break

                    # concatenate the relative path
                    image_path = os.path.join(image_sets_folder, image_path)

                    # check if the image path is valid
                    image_path_check = os.path.isfile(image_path)

                    if image_path_check is True:
                        self.img_list.append(cv.imread(image_path))
                        self.img_path_list.append(image_path)
                        valid_path = True
                        image_counter += 1
                    else:
                        raise IOError
                except IOError:
                    print('invalid image path for image (%s)' % (image_path))
