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

            # magnitude and theta for each pixel in the window
            self.magnitudes = np.zeros((16, 16), dtype=np.float32)
            self.thetas = np.zeros((16, 16), dtype=np.float32)

            # orientation histogram
            # 16x16 window consists of 16 4x4 cells, each cell orients one of
            # the 8 directions representing by a 8-dimensional vector
            self.orientation_histogram = np.zeros((16, 8), dtype=np.int)

            # normalized version of orientation histogram
            self.orientation_histogram_norm_128 = np.zeros((1, 128), dtype=np.float32)

    def normalize_descriptor(self):
        """
        normalize self.orientation_histogram_norm_128 such that sum of square
        of elements = 1 and each element < 0.2
        (but there is always a few element >= 0.2 slightly)
        """
        self.orientation_histogram_norm_128 = self.orientation_histogram_norm_128 / norm(self.orientation_histogram_norm_128)
        self.orientation_histogram_norm_128 = np.clip(self.orientation_histogram_norm_128, 0, 0.2)
        self.orientation_histogram_norm_128 = self.orientation_histogram_norm_128 / norm(self.orientation_histogram_norm_128)

    def flatten_orientation_histogram(self):
        """
        set self.self.orientation_histogram_norm_128
        """
        self.orientation_histogram_norm_128 = np.ndarray.flatten(self.orientation_histogram)

    def get_descriptor(self):
        """
        return flattened self.orientation_histogram
        :return: flattened self.orientation_histogram
        """
        return self.orientation_histogram_norm_128
