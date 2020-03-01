import numpy as np
import random
import cv2 as cv

class hom_calculator:

    def __init__(self, inlierThreshold, numIterations, dmatch_list,
        interest_point_match_image_1, interest_point_match_image_2):
        """
        constructor

        :param inlierThreshold: inlier threshold
        :param numIterations:   number of iterations of RANSAC
        :param dmatch_list:     a list of DMatch
        :param interest_point_match_image_1: interest point matching list in image 1
        :param interest_point_match_image_2: interest point matching list in image 2
        """
        # set threshold
        self.inlierThreshold = inlierThreshold

        # set iterations (epsilon in slide page 155)
        self.numIterations = numIterations

        # set matching points
        self.dmatch_list = dmatch_list
        # (index 0 in interest_point_match_image_1 <--> index 0 in interest_point_match_image_2)
        self.interest_point_match_image_1 = interest_point_match_image_1
        self.interest_point_match_image_2 = interest_point_match_image_2

        # set number of matches
        self.numMatches = len(interest_point_match_image_1)

        # initialize homography and inverse homography
        self.homography = np.empty((3, 3), dtype=np.float32)
        self.homographyInv = np.empty((3, 3), dtype=np.float32)

    def project(self, x1, y1, H, x2=0, y2=0):
        """
        project point(x1, y1) from image 1 to (x2, y2) in image 2

        :param x1: x value in image 1's point
        :param y1: y value in image 1's point
        :param H:  homography
        :param x2: projected x value in image 2's point (doesn't use)
        :param y2: projected y value in image 2's point (doesn't use)
        :return: (x2, y2)
        """
        denominator = H[2, 0] * x1 + H[2, 1] * y1 + H[2, 2] * 1
        numerator_x2 = H[0, 0] * x1 + H[0, 1] * y1 + H[0, 2] * 1
        numerator_y2 = H[1, 0] * x1 + H[1, 1] * y1 + H[1, 2] * 1

        if denominator == 0:
            # if denominator is exactly 0, then add a small epsilon to avoid
            # divide by 0
            denominator += 0.001

        x2 = numerator_x2 / denominator
        y2 = numerator_y2 / denominator

        return x2, y2

    def computeInlierCount(self, H, matches, numMatches, inlierThreshold):
        """
        a helper function for RANSAC that computes the number of inlying
        points given a homography H
        :param H:               homography of iter-th
        :param matches:         all matches (self.dmatch_list)
        :param numMatches:      number of matches (self.numMatches)
        :param inlierThreshold: inlier threshold (self.inlierThreshold)
        :return: the number of inlying points
        """

        inlier_count = 0

        # because I insert the interest points in two images into two lists
        # correspondingly (i.e. index k in self.interest_point_match_image_1
        # matches to index k in self.interest_point_match_image_2), I don't
        # need to use dmatch_list to index using _queryIdx, _trainIdx

        for i in range(numMatches):
            # src: (x1, y1) in image 1
            # target: (x2, y2) in image 2
            x1, y1 = self.interest_point_match_image_1[i].pt
            x2, y2 = self.interest_point_match_image_2[i].pt

            # projected x1' ad y1' on homography
            x1_prime, y1_prime = self.project(x1, y1, H)

            # calculate the residual error ||pi', H * pi ||
            # || . || represents root of sum of squares
            residual = np.sqrt((x2 - x1_prime) ** 2 + (y2 - y1_prime) ** 2)

            # compute inliers where residual < epsilon
            if residual < inlierThreshold:
                inlier_count += 1

        return inlier_count

    def find_all_inliers(self, best_homography):
        """
        after computing the best homography with highest no. of inliers,
        once again find all the inliers

        :param best_homography: best homography
        :return: all inlier matching and corresponding DMatch list
        """
        best_hom_match_ip_image_1 = list()
        best_hom_match_ip_image_2 = list()
        best_hom_dmatch_list = list()
        dmatch_index = 0

        for i in range(self.numMatches):
            # src: (x1, y1) in image 1
            # target: (x2, y2) in image 2
            x1, y1 = self.interest_point_match_image_1[i].pt
            x2, y2 = self.interest_point_match_image_2[i].pt

            # projected x1' ad y1' on the best homography
            x1_prime, y1_prime = self.project(x1, y1, best_homography)

            # calculate the residual error ||pi', H * pi ||
            # || . || represents root of sum of squares
            residual = np.sqrt((x2 - x1_prime) ** 2 + (y2 - y1_prime) ** 2)

            if residual < self.inlierThreshold:
                # append 2 interest points correspondingly
                best_hom_match_ip_image_1.append(self.interest_point_match_image_1[i])
                best_hom_match_ip_image_2.append(self.interest_point_match_image_2[i])

                # create a new DMatch object with different index, then append
                # it to best_hom_dmatch_list
                best_hom_inlier_dmatch = cv.DMatch(dmatch_index, dmatch_index, 2)
                best_hom_dmatch_list.append(best_hom_inlier_dmatch)

                dmatch_index += 1     # increment the index

        return best_hom_match_ip_image_1, best_hom_match_ip_image_2, best_hom_dmatch_list

    def RANSAC(self, matches, numIterations, inlierThreshold, hom=None,
            homInv=None, image1Display=None, image2Display=None):
        """
        takes a list of potentially matching points between two images and
        returns the homography transformation that relates them

        :param matches:             a DMatch list
        :param numIterations:       number of iterations of RANSAC
        :param inlierThreshold:     inlier threshold
        :param hom:                 homography
        :param homInv:              inverse of homography
        :param image1Display:
        :param image2Display:
        :return:
        """
        # initialize best parameters
        highest_inlier_count = 0
        best_homography = np.empty((3, 3), dtype=np.float32)

        # C.a. for "numIterations" iterations, do the following
        for iter in range(numIterations):
            # use iteration no. also to set random seed
            # (ensure in each iteration generating different 4 random numbers)
            random.seed(iter)

            # C.a.i. randomly selected 4 pairs of potentially matching points
            # from matches
            random_list = list()
            while len(random_list) < 4:
                random_num = random.randint(0, self.numMatches - 1)
                if random_num not in random_list:
                    random_list.append(random_num)
            # initialize placeholders
            random_4_interest_points_image_1 = np.empty((4, 2), dtype=np.float32)
            random_4_interest_points_image_2 = np.empty((4, 2), dtype=np.float32)
            # insert 4 random matching interest points
            for i in range(len(random_list)):
                random_4_interest_points_image_1[i, 0], random_4_interest_points_image_1[i, 1] = self.interest_point_match_image_1[random_list[i]].pt
                random_4_interest_points_image_2[i, 0], random_4_interest_points_image_2[i, 1] = self.interest_point_match_image_2[random_list[i]].pt

            # C.a.ii. compute the homography relating the four selected
            # matches with the function cv.findHomography(...)
            iter_th_homography, _ = cv.findHomography(
                srcPoints=random_4_interest_points_image_1,
                dstPoints=random_4_interest_points_image_2,
                method=0)

            # using the computed homography, compute the number of inliers
            # using "computeInlierCount"
            iter_th_inlier_count = self.computeInlierCount(
                H=iter_th_homography,
                matches=self.dmatch_list,
                numMatches=self.numMatches,
                inlierThreshold=inlierThreshold
            )

            # C.a.iii. if this homography produces the highest number of
            # inliers, store it as the best homography
            if iter_th_inlier_count > highest_inlier_count:
                highest_inlier_count = iter_th_inlier_count
                best_homography = iter_th_homography

        # set best homography
        self.homography = best_homography
        # set inverse of best homography
        # Attention: here should be pseudo-inverse!
        #            there are 2 ways to calculate inverse matrix:
        #               1. np.linalg.inv: real inverse
        #               2. np.linalg.pinv: pseudo inverse
#        self.homographyInv = np.linalg.inv(best_homography)
        self.homographyInv = np.linalg.pinv(best_homography)

        # C.b. Given the highest scoring homography, once again find all the
        # inliers. Compute a new refined homography using all of the inliers
        # find all the inliers
        best_hom_match_ip_image_1, best_hom_match_ip_image_2, best_hom_dmatch_list = self.find_all_inliers(best_homography)

        # C.c. Display the inlier matches using cv.drawMatches(...)
        # I apply this in main function
        return best_hom_match_ip_image_1, best_hom_match_ip_image_2, best_hom_dmatch_list
