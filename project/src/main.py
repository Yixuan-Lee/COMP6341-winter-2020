import cv2 as cv
from helper_functions import read_all_images
from helper_functions import read_threshold_harris
from helper_functions import save_image
from helper_functions import read_ssd_threshold
from helper_functions import read_ratio_test_threshold
from harris_corner_detector import harris_corner
from image_descriptors import descriptors
from descriptors_matcher import matcher
from filename_manager import fname_manager

# store the stitched image
stitched_image = None
# file name manager
name_manager = fname_manager()


def main():
    # read all images
    image_list = list()
    while len(image_list) == 0:
        image_list = read_all_images()

        if len(image_list) == 0:
            print('you have to enter at least 1 image.')
    # show all the reading images
#     for image in image_list:
#         cv.imshow('image', image)
#         cv.waitKey(0)
#         cv.destroyAllWindows()

    # read harris corner detection threshold
    harris_threshold = read_threshold_harris()

    # read ssd distance threshold and ratio test threshold used in matching
    ssd_threshold = read_ssd_threshold()
    ratio_test_threshold = read_ratio_test_threshold()

    if len(image_list) == 1:
        # if program only inputs 1 image, then save an result image called
        # '1a.png' showing the Harris response of the image under the folder
        # 'result_images/', then program exits (due to no enough images to do
        # image stitching)

        # instantiate a harris_corner instance
        harris = harris_corner(image_list[0])

        # do harris corner detection on the only 1 image
        harris.harris_corner_detection(harris_threshold)

        # save the harris response to 1a.png under the folder 'result_images/'
        # then show the image
        print('------- Saved Harris Response to 1a.png -------')
        save_image('1a.png', harris.R_normalized)

        print('------- Showing result_images/1a.png -------')
        cv.imshow('Harris response --> 1a.png', harris.R_normalized)
        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        # if program inputs more than 1 image, then
        #
        # step 1: save result images of the detected corners of all images.
        #         called '1b.png', '1c.png', '1d.png', ... under the folder
        #         '../result_images/'
        #
        # step 2: matching the interest points between 2 images,
        #         save the image showing the image with the found matches
        #         called '2b.png', '2c.png', '2d.png', ... under the folder
        #         '../result_images/'
        #
        # step 3: save the images after RANSAC (should only contain inliers)
        #         '3b.png', '3c.png', '3d.png', ... under the folder
        #         '../result_images/'
        #
        # step 4: save the stitched image as '4a.png', '4b.png', ... under
        #         the folder '../result_images/'
        #

        ############################ Main Loop ############################
        while len(image_list) > 0:
            if stitched_image == None:
                # indicates this is the first time in the loop
                image_1 = image_list.pop(0)     # index-0 image
                image_2 = image_list.pop(0)     # index-1 image before popping
            else:
                # indicates this is NOT the first time in the loop
                image_1 = stitched_image[0]     # previously stitched image
                image_2 = image_list.pop(0)     # index-0 in image_list

            # ################# Step 1. Feature Detection ################# #
            print('-------------- Feature Detection --------------')
            print()
            # instantiate 2 harris_corner instances
            harris_1 = harris_corner(image_1)
            harris_2 = harris_corner(image_2)
            # do harris corner detection
            harris_1.harris_corner_detection(harris_threshold)
            harris_2.harris_corner_detection(harris_threshold)
            print('------------ Feature Detection Done ------------')
            print()

            # get the filenames for corner detection saving files
            (harris_save_fname_1, harris_save_fname_2) = name_manager.get_2_harris_output_filenames()
            print('------ Saving detected corners to files (%s), (%s) ------' % (harris_save_fname_1, harris_save_fname_2))
            print()
            harris_1_out = image_1.copy()
            harris_2_out = image_2.copy()
            cv.drawKeypoints(
                image=image_1,
                keypoints=harris_1.interest_points_list,
                outImage=harris_1_out,
                color=(255, 0, 0)
            )
            cv.drawKeypoints(
                image=image_2,
                keypoints=harris_2.interest_points_list,
                outImage=harris_2_out,
                color=(255, 0, 0)
            )
            save_image(harris_save_fname_1, harris_1_out)
            save_image(harris_save_fname_2, harris_2_out)
            print('------------ Saving detected corners Done ------------')
            print()

            print('------- Showing detected corners result images  -------')
            print()
            cv.imshow('Detected corners -> ' + harris_save_fname_1, harris_1_out)
            cv.imshow('Detected corners -> ' + harris_save_fname_2, harris_2_out)
            cv.waitKey(0)
            cv.destroyAllWindows()

            # ################# Step 2. Feature Matching ################# #
            print('-------------- Feature Matching --------------')
            print()
            # instantiate 2 descriptors instances to hold all ip descriptors
            descriptors_1 = descriptors(image_1, harris_1.interest_points_list)
            descriptors_2 = descriptors(image_2, harris_2.interest_points_list)

            # calculate feature descriptors for all interest points
            descriptors_1.calc_sift_descriptor_for_all_interest_points()
            descriptors_2.calc_sift_descriptor_for_all_interest_points()

            # instantiate a matcher instance
            matcher_1_2 = matcher(descriptors_1, descriptors_2, ssd_threshold, ratio_test_threshold)
            # match the descriptors in two images
            matcher_1_2.matching_descriptors()

            match_out = cv.drawMatches(
                img1=image_1,
                keypoints1=matcher_1_2.interest_point_match_image_1,
                img2 = image_2,
                keypoints2=matcher_1_2.interest_point_match_image_2,
                matches1to2=matcher_1_2.dmatch_list,
                outImg=None
            )
            print('------------ Feature Matching Done ------------')
            print()

            # get the filename for saving matching image
            matching_save_file_name = name_manager.get_matching_output_filename()
            print('-------- Saving matching result to (%s) --------' % (matching_save_file_name))
            save_image(matching_save_file_name, match_out)
            print('------------ Saving detected corners Done ------------')
            print()

            print('------------ Showing matching result image ------------')
            print()
            cv.imshow('Feature matching -> ' + matching_save_file_name, match_out)
            cv.waitKey(0)
            cv.destroyAllWindows()

            # ################# Step 3.  ################# #

            # ################# Step 4.  ################# #


if __name__ == '__main__':
    main()
