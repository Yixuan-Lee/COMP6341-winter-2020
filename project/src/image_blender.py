import cv2 as cv
import numpy as np

class image_stitch:

    def __init__(self, image1, image2, homography_calculator):
        """
        constructor

        :param image1: image 1
        :param image2: image 2
        :param homography_calculator:   hom_calculator instance (for calling project(.) function)
        """
        self.image_1 = image1
        self.image_2 = image2
        self.homography = homography_calculator.homography
        self.homographyInv = homography_calculator.homographyInv
        self.homo_calculator = homography_calculator

    def stitch(self, image1, image2, hom=None, homInv=None, stithcedImage=None):
        """

        :param image1:          image 1
        :param image2:          image 2
        :param hom:             homography (use self.homography)
        :param homInv:          homography inverse (use self.homographyInv)
        :param stithcedImage:   stitched image placeholder (doesn't use)
        :return: stitched image
        """
        # A.a. Compute the size of stitched image
        # A.a.i. get the 4 corners of image 2
        # image2.shape = (# of rows, # of columns, # of channels)
        max_rows_2, max_columns_2, _ = image2.shape # omit the 3rd parameter (which is # of channels)
        # left top (0, 0)
        x_left_top_2 = 0
        y_left_top_2 = 0

        # right top (max_columns, 0)
        x_right_top_2 = max_columns_2
        y_right_top_2 = 0

        # left bottom (0, max_rows)
        x_left_bottom_2 = 0
        y_left_bottom_2 = max_rows_2

        # right bottom (max_columns, max_rows)
        x_right_bottom_2 = max_columns_2
        y_right_bottom_2 = max_rows_2

        # A.a.ii. project 4 corners onto image1 using project(...) and homInv
        x_left_top_2_projected, y_left_top_2_projected = self.homo_calculator.project(x_left_top_2, y_left_top_2, self.homographyInv)
        x_right_top_2_projected, y_right_top_2_projected = self.homo_calculator.project(x_right_top_2, y_right_top_2, self.homographyInv)
        x_left_bottom_2_projected, y_left_bottom_2_projected = self.homo_calculator.project(x_left_bottom_2, y_left_bottom_2, self.homographyInv)
        x_right_bottom_2_projected, y_right_bottom_2_projected = self.homo_calculator.project(x_right_bottom_2, y_right_bottom_2, self.homographyInv)

        # A.a.iii. get 4 corners in image 1
        max_rows_1, max_columns_1, _ = image1.shape # omit the 3rd parameter (which is # of channels)
        x_left_top_1 = 0
        y_left_top_1 = 0
        x_right_top_1 = max_columns_1
        y_right_top_1 = 0
        x_left_bottom_1 = 0
        y_left_bottom_1 = max_rows_1
        x_right_bottom_1 = max_columns_1
        y_right_bottom_1 = max_rows_1

        # A.a.iv. find x_min, x_max, y_min, y_max based on 4 projected points + 4 corners in image 1
        x_min, x_max, y_min, y_max = self.findStitchedImageSize(
            x_coordinates=[x_left_top_1, x_right_top_1, x_left_bottom_1, x_right_bottom_1, x_left_top_2_projected, x_right_top_2_projected, x_left_bottom_2_projected, x_right_bottom_2_projected],
            y_coordinates=[y_left_top_1, y_right_top_1, y_left_bottom_1, y_right_bottom_1, y_left_top_2_projected, y_right_top_2_projected, y_left_bottom_2_projected, y_right_bottom_2_projected]
        )

        # A.b. copy image1 onto the stitchedImage at the right location
        # initialized stitchedImage
        stitched = np.zeros((int(y_max - y_min), int(x_max - x_min), 3), dtype=np.float32)
        # copy image1 onto the stitchedImage
        x_offset = int(0 - x_min)
        y_offset = int(0 - y_min)
        stitchedImage_1 = self.copyImageOne(x_offset, y_offset, stitched)

        # A.c. For each pixel in stitchedImage, project the point onto
        #      "image2", if it lies withing image2's boundaries, add or blend
        #      the pixel's value to stitchedImage
        stitchedImage_1_2 = self.copyImageTwo(x_offset, y_offset, stitchedImage_1)

        # record the final version of stitched image of image 1 and 2 (dtype = np.float32)
        self.stitchedImage = stitchedImage_1_2

        # convert np.float32 -> int8 (for image presentation), then return
        return cv.convertScaleAbs(stitchedImage_1_2)

    def findStitchedImageSize(self, x_coordinates, y_coordinates):
        """
        find x_min, x_max, y_min, y_max

        :param x_coordinates: coordinate values on x-axis
        :param y_coordinates: coordinate values on y-axis
        :return: x_min, x_max, y_min, y_max
        """
        x_min, x_max = min(x_coordinates), max(x_coordinates)
        y_min, y_max = min(y_coordinates), max(y_coordinates)

        return x_min, x_max, y_min, y_max

    def copyImageOne(self, x_offset, y_offset, stitchedImage):
        """
        copy image 1 pixels onto the stitchedImage

        :param x_offset:        x offset
        :param y_offset:        y offset
        :param stitchedImage:   stitched image
        """
        max_rows, max_columns, _ = self.image_1.shape

        for i in range(max_rows):           # i -> y
            for j in range(max_columns):    # j -> x
                stitchedImage[i + y_offset, j + x_offset, :] = self.image_1[i, j, :]

        return stitchedImage

    def copyImageTwo(self, x_offset, y_offset, stitchedImage):
        """
        copy image 2 pixels onto the stitchedImage

        :param x_offset:        x offset
        :param y_offset:        y offset
        :param stitchedImage:   stitched image
        :return numpy array of blended stitched image
        """
        # A.c.i. For each pixel in stitchedImage, project the point onto
        #        image2
        max_rows, max_columns, _ = stitchedImage.shape
        blended_stitchedImage = np.empty(stitchedImage.shape, dtype=np.float32)
        for i in range(max_rows):           # i -> y
            for j in range(max_columns):    # j -> x
                # relative coordinates to image 1 index in stitched image
                x_val_in_stitchedImage = j - x_offset
                y_val_in_stitchedImage = i - y_offset
                # project the relative coordinates onto image 2
                x_val_in_image_2, y_val_in_image_2 = self.homo_calculator.project(x_val_in_stitchedImage, y_val_in_stitchedImage, self.homography)

                # make the x_val_in_image_2, y_val_in_image_2 integers
                # Question: should we have to use cv.getRectSubPix(.) in the project ?
                x_val_in_image_2 = int(x_val_in_image_2)
                y_val_in_image_2 = int(y_val_in_image_2)

                # check if point (x_val_in_stitchedImage, y_val_in_stitchedImage) in image_1 and image_2
                is_in_image_1 = self.isInImage1(x_val_in_stitchedImage, y_val_in_stitchedImage)
                is_in_image_2 = self.isInImage2(x_val_in_image_2, y_val_in_image_2)

                if is_in_image_2 and not is_in_image_1:
                    # if (x_val_in_stitchedImage, y_val_in_stitchedImage) lies within image2's boundaries
                    # but not within image1's boundaries, then we just copy the image2's pixel B/G/R values
                    # from self.image_2 -> blended_stitchedImage
                    blended_stitchedImage[i, j, :] = self.image_2[y_val_in_image_2, x_val_in_image_2, :]
                elif is_in_image_2 and is_in_image_1:
                    # A.c.ii. if it lies within the intersection of image2' boundaries and image1' boundaries, then
                    # add or blend the pixel's value to stitchedImage
                    # Here, I applied alpha blending between the pixel stitchedImage[i, j, :] and
                    # self.image_2[x_val_in_image_2, y_val_in_image_2, :] with equal weights
                    alpha_stitched = 0.5
                    alpha_image2 = 0.5

                    # blend the pixel on blue/green/red (formulas on page 17, slide lecture 9: Image Stitching II)
                    blended_stitchedImage[i, j, :] = (alpha_stitched * stitchedImage[i, j, :] + alpha_image2 * self.image_2[y_val_in_image_2, x_val_in_image_2, :]) / (alpha_stitched + alpha_image2)
                elif is_in_image_1 and not is_in_image_2:
                    # if (x_val_in_stitchedImage, y_val_in_stitchedImage) lies within image1's boundaries
                    # but not within image2's boundaries, then copy the stitchedImage -> blended_stitchedImage
                    blended_stitchedImage[i, j, :] = stitchedImage[i, j, :]
                else:
                    # if we go in here, it means pixel (x_val_in_stitchedImage, y_val_in_stitchedImage) in stitched image
                    # goes outside of image_1 and image_2  ->  black area
                    # set black pixel
                    blended_stitchedImage[i, j, :] = np.array([0, 0, 0])

        return blended_stitchedImage

    def isInImage1(self, x, y):
        """
        check if (x, y) is in image 1

        :param x: x coordinate
        :param y: y coordinate
        :return: true if it is in, false otherwise
        """
        max_rows, max_columns, _ = self.image_1.shape

        if x >= 0 and x < max_columns and y >= 0 and y < max_rows:
            return True

        return False

    def isInImage2(self, x, y):
        """
        check if (x, y) is in image 2

        :param x: x coordinate
        :param y: y coordinate
        :return: true if it is in, false otherwise
        """
        max_rows, max_columns, _ = self.image_2.shape

        if x >= 0 and x < max_columns and y >= 0 and y < max_rows:
            return True

        return False

