import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


# global variables
image_src = '../image_set'
bmp_images = (
    'crayons_mosaic.bmp',
    'oldwell_mosaic.bmp',
    'pencils_mosaic.bmp')
jpg_images = (
    'crayons.jpg',
    'oldwell.jpg',
    'pencils.jpg')


def read_images():
    """
    read BMP and JPEG images then return

    :return: BMP and JPEG images
    """
    image_index = -1

    while image_index == -1:
        try:
            image_index = int(input('image_index = '))

            if image_index < 0 or image_index >= len(bmp_images):
                raise ValueError

        except ValueError:
            # IndexOutofBound of TypeError
            print('Message: image_index should be input as an integer '
                  'between 0 and', len(bmp_images))
            image_index = -1

    # read JPG image as original
    original = cv2.imread(
        os.path.join(image_src, jpg_images[image_index]),
        cv2.IMREAD_COLOR)

    # read BMP image as bayer
    bayer = cv2.imread(
        os.path.join(image_src, bmp_images[image_index]),
        cv2.IMREAD_GRAYSCALE)

    return original, bayer


def separate_channels(bayer_image):
    """
    separate 3 (B, G, R) channels for bayer image then return

    :param bayer_image: bayer image
    :return: blue_channel, green_channel, red_channel
    """
    height, width = bayer_image.shape
    blue_channel, green_channel, red_channel = np.zeros((height, width)), np.zeros((height, width)), np.zeros((height, width))

    # set pixels to corresponding channel
    for row in range(height):
        for col in range(width):
            if row % 2 == 0 and col % 2 == 0:
                blue_channel[row, col] = bayer_image[row, col]
            elif row % 2 == 1 and col % 2 == 1:
                green_channel[row, col] = bayer_image[row, col]
            elif (row % 2 == 0 and col % 2 == 1) or (row % 2 == 1 and col % 2 == 0):
                red_channel[row, col] = bayer_image[row, col]

    return blue_channel, green_channel, red_channel


def get_channel_kernels():
    """
    define and return the kernel for each channel

    :return: blue_kernel, green_kernel, red_kernel
    """
    blue_kernel = np.array([
        [1/4, 1/2, 1/4],
        [1/2, 1,   1/2],
        [1/4, 1/2, 1/4]], dtype=np.float32)
    green_kernel = np.array([
        [1/4, 1/2, 1/4],
        [1/2, 1,   1/2],
        [1/4, 1/2, 1/4]], dtype=np.float32)
    red_kernel = np.array([
        [0,   1/4, 0],
        [1/4, 1,   1/4],
        [0,   1/4, 0]], dtype=np.float32)

    return blue_kernel, green_kernel, red_kernel


def get_demosaic_channels(blue_channel, green_channel, red_channel, blue_kernel, green_kernel, red_kernel):
    """
    demosaicing 3 channels then return

    :return: demosaic_blue_channel, demosaic_green_channel, demosaic_red_channel
    """
    # 2D convolution
    demosaic_blue_channel = cv2.filter2D(src=blue_channel, ddepth=-1, kernel=blue_kernel)
    demosaic_green_channel = cv2.filter2D(src=green_channel, ddepth=-1, kernel=green_kernel)
    demosaic_red_channel = cv2.filter2D(src=red_channel, ddepth=-1, kernel=red_kernel)

    return demosaic_blue_channel, demosaic_green_channel, demosaic_red_channel


def get_squared_differences(original, demosaic):
    """
    compute the squared differences between original and demosaic over 3 color channels

    :param original: original image from JPG image
    :param demosaic: demosaic image from BMP image
    :return: squared differences of two images
    """
    return np.square(original - demosaic)


def part_one_show(original, demosaic_blue_channel, demosaic_green_channel, demosaic_red_channel):
    """
    show the 3 images output described in guideline Part 1

    :param original:                JPG image (shape of (height, width, no. of channels))
    :param demosaic_blue_channel:   demosaic blue channel
    :param demosaic_green_channel:  demosaic green channel
    :param demosaic_red_channel:    demosaic red channel
    """
    # stack B/G/R channels together
    demosaic = np.dstack((demosaic_blue_channel, demosaic_green_channel, demosaic_red_channel))

    # compute the squared differences between the original and the demosaic
    squared_diff = get_squared_differences(original, demosaic)

    # important for cv2.imshow!!
    # convert to unsigned int8 type so that the cv2.imshow will show using
    # range [0, 255] instead of [0, 1]
    demosaic = demosaic.astype(np.uint8)
    squared_diff = squared_diff.astype(np.uint8)

    # concatenate the 3 images horizontally
    concat = np.hstack((original, demosaic, squared_diff))

    cv2.imshow(winname='part 1 output', mat=concat)
    # cv2.imshow('original', original)
    # cv2.imshow('demosaic', demosaic)
    # cv2.imshow('squared difference', squared_diff)
    cv2.waitKey(0)


def main():
    """
    main function
    """
    # read the images
    original, bayer = read_images()

    # debug: showing the 2 images
    # cv2.imshow('original', original)
    # cv2.imshow('bayer', bayer)
    # cv2.waitKey(0)

    # debug: showing the type and shapes
    # print(type(original))  # <class 'numpy.ndarray'>
    # print(original.shape)  # (Height, Width, # of Channels)
    # print(bayer.shape)

    # separate channels for bayer image
    blue_channel, green_channel, red_channel = separate_channels(bayer)

    # initialize kernel for each channel
    blue_kernel, green_kernel, red_kernel = get_channel_kernels()

    # demosaicing
    demosaic_blue_channel, demosaic_green_channel, demosaic_red_channel = \
        get_demosaic_channels(
            blue_channel, green_channel, red_channel,
            blue_kernel, green_kernel, red_kernel)

    # show the concatenated 3 output images in part 1
    part_one_show(original, demosaic_blue_channel, demosaic_green_channel,
        demosaic_red_channel)


if __name__ == '__main__':
    main()
