import cv2 as cv
from helper_functions import print_params_table
from helper_functions import delete_all_previous_result_images
from helper_functions import read_threshold_harris
from helper_functions import save_image
from helper_functions import read_ssd_threshold
from helper_functions import read_ratio_test_threshold
from helper_functions import read_inlier_threshold
from helper_functions import read_no_of_iterations
from my_images import my_image_list
from harris_corner_detector import harris_corner
from image_descriptors import descriptors
from descriptors_matcher import matcher
from filename_manager import fname_manager
from homography_calculator import hom_calculator
from image_blender import image_stitch

# store the stitched image
stitched_image = None
# file name manager
name_manager = fname_manager()


def main():
    # read all images
    global stitched_image
    image_list = my_image_list()
    while image_list.length() == 0:
        image_list.read_all_images()

        if image_list.length() == 0:
            print('you have to enter at least 1 image.')

    # read harris corner detection threshold
    harris_threshold = read_threshold_harris()

    # read ssd distance threshold and ratio test threshold used in matching
    ssd_threshold = read_ssd_threshold()
    ratio_test_threshold = read_ratio_test_threshold()

    # read inlier threshold and number of iterations used to find homography
    inlier_threshold = read_inlier_threshold()
    number_of_iterations = read_no_of_iterations()

    # print all used parameters table
    print_params_table(image_list, harris_threshold, ssd_threshold,
        ratio_test_threshold, inlier_threshold, number_of_iterations)

    # delete all previous resulting images under '../result_images' folder
    delete_all_previous_result_images()

    if image_list.length() == 1:
        # if the program only inputs 1 image, then save an result image called
        # '1a.png' showing the Harris response of the image under the folder
        # 'result_images/', then program exits (due to no enough image to do
        # image stitching)

        # instantiate a harris_corner instance
        harris = harris_corner(image_list.img_list[0])

        # do harris corner detection on the only 1 image
        harris.harris_corner_detection(harris_threshold)

        # save the harris response to 1a.png under the folder 'result_images/'
        # then show the image
        print('------- Saved Harris Response to 1a.png -------')
        print()
        save_image('1a.png', harris.R_normalized)

        print('------- Showing result_images/1a.png -------')
        print()
        cv.imshow('Harris response --> 1a.png', harris.R_normalized)
        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        # if program inputs more than 1 image, then
        #
        # step 1: save result images of the detected corners of all images
        #         called '1b.png', '1c.png', '1d.png', ... under the folder
        #         '../result_images/'
        #
        # step 2: matching the interest points between 2 images,
        #         save the image showing the image with the found matches
        #         called '2b.png', '2c.png', '2d.png', ... under the folder
        #         '../result_images/'
        #
        # step 3: save the matching images after RANSAC (should only contain
        #         inlier matches) called '3b.png', '3c.png', '3d.png', ...
        #         under the folder '../result_images/'
        #         ('3b.png' corresponds to '2b.png', etc.)
        #
        # step 4: save the stitched image as '4b.png', '4c.png', ... under
        #         the folder '../result_images/'
        #

        ############################ Main Loop ############################
        while image_list.length() > 0:
            if stitched_image is None:
                # indicates this is the first time in the loop
                image_1 = image_list.img_list.pop(0)    # index-0 image
                image_2 = image_list.img_list.pop(0)    # index-1 image before popping above
            else:
                # indicates this is NOT the first time in the loop
                image_1 = stitched_image                # previously stitched image
                image_2 = image_list.img_list.pop(0)    # index-0 in image_list

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
            print()
            save_image(matching_save_file_name, match_out)
            print('------------ Saving detected corners Done ------------')
            print()

            print('------------ Showing matching result image ------------')
            print()
            cv.imshow('Feature matching -> ' + matching_save_file_name, match_out)
            cv.waitKey(0)
            cv.destroyAllWindows()

            # ################# Step 3. Mosaic Stitching  ################# #
            print('--------------- Running RANSAC ---------------')
            print()
            # instantiate a RANSAC instance
            homography = hom_calculator(inlier_threshold,
                number_of_iterations, matcher_1_2.dmatch_list,
                matcher_1_2.interest_point_match_image_1,
                matcher_1_2.interest_point_match_image_2)
            # compute the best homography and reserve all good matches
            best_hom_match_ip_image_1, best_hom_match_ip_image_2, best_hom_dmatch_list = homography.RANSAC(
                matches=matcher_1_2.dmatch_list,
                numIterations=number_of_iterations,
                inlierThreshold=inlier_threshold
            )
            # this should only contain inliers (i.e. only good matches)
            RANSAC_match_out = cv.drawMatches(
                img1=image_1,
                keypoints1=best_hom_match_ip_image_1,
                img2=image_2,
                keypoints2=best_hom_match_ip_image_2,
                matches1to2=best_hom_dmatch_list,
                outImg=None
            )
            print('------------ Running RANSAC Done ------------')
            print()

            # get the filename for saving RANSAC matching
            RANSAC_matching_save_file_name = name_manager.get_RANSAC_matching_output_filename()
            print('-------- Saving matching result after RANSAC to (%s) --------' % (RANSAC_matching_save_file_name))
            print()
            save_image(RANSAC_matching_save_file_name, RANSAC_match_out)
            print('------------ Saving matching result after RANSAC Done ------------')
            print()

            print('---------- Drawing results of RANSAC ----------')
            print()
            cv.imshow('matches after RANSAC -> ' + RANSAC_matching_save_file_name, RANSAC_match_out)
            cv.waitKey(0)
            cv.destroyAllWindows()
            print('-------- Drawing results of RANSAC Done --------')
            print()

            # ################# Step 4. Image Stitching ################# #
            print('--------------- Blending 2 images ---------------')
            print()
            # instantiate an image blender instance
            blender = image_stitch(image_1, image_2, homography)
            # blend 2 images
            stitched = blender.stitch(image_1, image_2)
            print('------------- Blending 2 images Done -------------')
            print()

            # get the filename for saving stitched image
            stitched_image_save_file_name = name_manager.get_image_stitch_output_filename()
            print('---------- Saving stitched image to (%s) ----------' % (stitched_image_save_file_name))
            print()
            save_image(stitched_image_save_file_name, stitched)
            print('------------ Saving stitched image Done ------------')
            print()

            print('--------- Drawing result after image stitching ---------')
            print()
            cv.imshow('stitched image -> ' + stitched_image_save_file_name, stitched)
            cv.waitKey(0)
            cv.destroyAllWindows()
            print('------- Drawing result after image stitching Done -------')
            print()

            # set stitched_image
            stitched_image = stitched


if __name__ == '__main__':
    main()
