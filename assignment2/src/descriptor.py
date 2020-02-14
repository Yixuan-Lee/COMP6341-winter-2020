import numpy as np
from numpy.linalg import norm


class sift_descriptor:

    def __init__(self, interest_point=None):
        """
        constructor of SIFT descriptor

        :param interest_point:  the interest point
        """
        if interest_point is not None:
            self.x = int(interest_point.pt[0])
            self.y = int(interest_point.pt[1])
            self.interest_point = interest_point

            # surrounding 18x18 window of neighborhood
            # the reason why here is not 16x16 is because when calculating
            # the magnitudes, it needs L(x+1, y), L(x-1, y), L(x, y+1),
            # L(x, y-1), so we have to have 1 padding for the border pixels
            self.window_18_18 = np.zeros((18, 18), dtype=np.float32)

            # magnitude and theta for each pixel in the 16x16 window
            self.magnitudes = np.zeros((16, 16), dtype=np.float32)
            self.dominant_magnitude = 0
            self.thetas = np.zeros((16, 16), dtype=np.float32)
            self.dominant_theta = 0

            # orientation histogram
            # 16x16 window consists of 16 4x4 cells, each cell orients one of
            # the 8 directions representing by a 8-dimensional vector
            self.orientation_histogram = np.zeros((16, 8), dtype=np.int)

            # normalized version of orientation histogram
            self.orientation_histogram_norm_128 = np.zeros((1, 128),
                dtype=np.float32)

    def set_magnitude_and_theta(self):
        """
        set magnitude and theta for each pixel in 16x16 window
        """
        ############## 2.1 angle descriptor ##############
        for i in range(16):
            for j in range(16):
                # set the indices which fit in descriptor.window_18_18 due to
                # the 1 extra padding
                w_18_18_i = i + 1
                w_18_18_j = j + 1

                # set the magnitude at descriptor.magnitudes[i, j] and
                # the theta at descriptor.thetas[i, j]
                diff_x = self.window_18_18[w_18_18_i + 1, w_18_18_j] - self.window_18_18[w_18_18_i - 1, w_18_18_j]
                diff_y = self.window_18_18[w_18_18_i, w_18_18_j + 1] - self.window_18_18[w_18_18_i, w_18_18_j - 1]

                # set magnitude
                magnitude = np.sqrt(diff_x ** 2 + diff_y ** 2)
                self.magnitudes[i, j] = magnitude

                # set theta
                theta = (np.arctan2(diff_x, diff_y) + np.pi) * 180 / np.pi
                self.thetas[i, j] = theta

                # set dominant_maginitude and dominant_theta


    def set_orientation_histogram_for_grid_cells(self):
        """
        calculate the orientation histogram for 16 4x4 grid cells
        """
        ############## 2.2 vote descriptor ##############
        window_size = 16
        grid_cell_idx = 0
        for i in range(0, window_size, 4):
            for j in range(0, window_size, 4):
                # orientation histogram of each cell
                grid_cell_orientations = np.zeros((1, 8), dtype=np.int32)

                for x in range(4):
                    for y in range(4):
                        # x and y coordinates in magnitudes and thetas
                        p_x = x + i
                        p_y = y + j

                        ############## 2.4 rotational invariant ##############
                        # compute the vote for which bin
                        vote = int(self.thetas[p_x, p_y] / 45)
                        if vote == 8:
                            # when descriptor.thetas[p_x, p_y] = 360, vote = 8
                            # the pixel votes for 7th bin
                            vote = 7

                        # increment the votes in the bin
                        grid_cell_orientations[0, vote] += 1

                # set the orientation histogram of the 4x4 cell to the
                # descriptor
                self.orientation_histogram[grid_cell_idx, :] = grid_cell_orientations
                grid_cell_idx += 1

        # flatten the orientation histogram so that to
        # set self.orientation_histogram_norm_128
        self.orientation_histogram_norm_128 = np.ndarray.flatten(
            self.orientation_histogram)

    def normalize_descriptor(self):
        """
        normalize self.orientation_histogram_norm_128 such that sum of square
        of elements = 1 and each element < 0.2
        (but there is always a few elements >= 0.2 slightly, if I make it in
        a while loop, there will be an infinite loop)
        """
        ############## 2.3 contrast invariance ##############
        self.orientation_histogram_norm_128 = self.orientation_histogram_norm_128 / norm(self.orientation_histogram_norm_128)
        self.orientation_histogram_norm_128 = np.clip(self.orientation_histogram_norm_128, 0, 0.2)
        self.orientation_histogram_norm_128 = self.orientation_histogram_norm_128 / norm(self.orientation_histogram_norm_128)

    def get_descriptor(self):
        """
        return flattened self.orientation_histogram, which is
        self.orientation_histogram_norm_128
        :return: flattened self.orientation_histogram
        """
        return self.orientation_histogram_norm_128
