"""
Comment:
Most of the code here are referenced to my assignment 2 implementation in
'feature_matching.py'
"""

import numpy as np
import cv2 as cv


class matcher:

    def __init__(self, descriptors_1, descriptors_2, ssd_threshold,
            ratio_test_threshold):
        """
        constructor

        :param descriptor_1: image_descriptors instance of image 1
        :param descriptor_2: image_descriptors instance of image 2
        :param ssd_threshold:           SSD threshold
        :param ratio_test_threshold:    ratio test threshold
        """
        # set image_descriptors
        self.image_descriptors_1 = descriptors_1
        self.image_descriptors_2 = descriptors_2

        # set thresholds
        self.ssd_threshold = ssd_threshold
        self.ratio_test_threshold = ratio_test_threshold

        # set interest point match lists
        self.interest_point_match_image_1 = list()
        self.interest_point_match_image_2 = list()

        # set DMatch result list
        self.dmatch_list = list()

    def feature_distance(self, descriptor_1, descriptor_2):
        """
        compute the sum of difference between 2 descriptors

        :return: sum of difference between 2 descriptors
        """
        return np.sum(np.square(descriptor_1.get_descriptor() -
                                descriptor_2.get_descriptor()))

    def matching_descriptors(self):
        """
        find the matching and store the DMatch in dmatch_list
        """
        # initialize variables
        best_ip_1 = None
        best_ip_2 = None
        ip_match_idx = 0

        for ip_1, descriptor_1 in self.image_descriptors_1.\
                interest_point_descriptor_dict.items():
            best_ssd_dist = float('inf')
            second_best_ssd_dist = float('inf')

            # SSD distance
            for ip_2, descriptor_2 in self.image_descriptors_2.\
                    interest_point_descriptor_dict.items():
                # compute the SSD distance between 2 descriptors
                ssd_dist = self.feature_distance(descriptor_1, descriptor_2)

                # filter out the SSD distance which is > the threshold
                if ssd_dist >= self.ssd_threshold:
                    continue

                if ssd_dist < best_ssd_dist:
                    # set best -> second_best
                    second_best_ssd_dist = best_ssd_dist

                    # set current -> best
                    best_ssd_dist = ssd_dist
                    best_ip_1 = ip_1
                    best_ip_2 = ip_2
                elif ssd_dist < second_best_ssd_dist:
                    second_best_ssd_dist = ssd_dist

            # ratio test
            # filter matches by comparing the ratio
            ratio = best_ssd_dist / second_best_ssd_dist
            if ratio > self.ratio_test_threshold:
                # meaning best and second best are very similar, so we discard
                # the match (associate with the 'fence' example in slide)
                continue
            else:
                # match corresponding feature descriptors
                # append the matched interest points in image_1 and image_2 to
                # 2 lists correspondingly
                self.interest_point_match_image_1.append(best_ip_1)
                self.interest_point_match_image_2.append(best_ip_2)

                # append the match to the match list
                match = cv.DMatch(ip_match_idx, ip_match_idx, 2)
                self.dmatch_list.append(match)

                ip_match_idx += 1
