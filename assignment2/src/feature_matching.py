import numpy as np
import cv2 as cv


def feature_distance(descriptor_1, descriptor_2):
    """
    compute the sum of difference between 2 descriptors

    :param descriptor_1: descriptor of interest point 1
    :param descriptor_2: descriptor of interest point 2
    :return: sum of difference between 2 descriptors
    """
    return np.sum(np.square(descriptor_1.get_descriptor() - descriptor_2.get_descriptor()))


def descriptors_matching(descriptor_dict_image_1, descriptor_dict_image_2,
        dmatch_list, ip_match_image_1, ip_match_image_2,
        ssd_threshold, ratio_test):
    """
    find the matching and store the DMatch in dmatch_list

    :param descriptor_dict_image_1: descriptor dictionary of image 1
    :param descriptor_dict_image_2: descriptor dictionary of image 2
    :param dmatch_list:             a list which stores DMatches
    :param ip_match_image_1:        a list which stores matched interest
                                    points in image 1
    :param ip_match_image_2:        a list which stores matched interest
                                    points in image 2
    :param ssd_threshold:           SSD distance threshold
    :param ratio_test:              ratio threshold (best/second_best)
    """
    # initialize variables
    best_ip_1 = None
    best_ip_2 = None
    ip_match_idx = 0

    for ip_1, descriptor_1 in descriptor_dict_image_1.items():
        best_ssd_dist = float('inf')
        second_best_ssd_dist = float('inf')

        ############## 3.2 SSD distance ##############
        for ip_2, descriptor_2 in descriptor_dict_image_2.items():
            # compute the SSD distance between 2 descriptors
            ssd_dist = feature_distance(descriptor_1, descriptor_2)

            # filter out the SSD distance which is bigger than the threshold
            if ssd_dist >= ssd_threshold:
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

        ############## 3.3 ratio test ##############
        # filter matches by comparing the ratio
        ratio = best_ssd_dist / second_best_ssd_dist
        if ratio > ratio_test:
            # meaning best and second best are very similar, so we discard
            # the match (associate with the 'fence' example in slide)
            continue
        else:
            ########## 3.1 match corresponding feature descriptors ##########
            # append the matched interest points in image_1 and image_2 to
            # 2 lists correspondingly
            ip_match_image_1.append(best_ip_1)
            ip_match_image_2.append(best_ip_2)

            # append the match to the match list
            match = cv.DMatch(ip_match_idx, ip_match_idx, 2)
            dmatch_list.append(match)

            ip_match_idx += 1
