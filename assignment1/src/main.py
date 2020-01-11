import os
import cv2
import numpy as np


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
            else:
                red_channel[row, col] = bayer_image[row, col]

    return blue_channel, green_channel, red_channel


def get_channel_kernels():
    """
    define and return the kernel for each channel

    :return: blue_kernel, green_kernel, red_kernel
    """
    blue_kernel = np.array([
        [1/4, 1/2, 1/4],
        [1/2, 1  , 1/2],
        [1/4, 1/2, 1/4]])
    green_kernel = np.array([
        [1/4, 1/2, 1/4],
        [1/2, 1  , 1/2],
        [1/4, 1/2, 1/4]])
    red_kernel = np.array([
        [0  , 1/4, 0  ],
        [1/4, 0  , 1/4],
        [0  , 1/4, 0  ]])

    return blue_kernel, green_kernel, red_kernel


def get_demosaic_channels(bayer_image, blue_channel, green_channel, red_channel, blue_kernel, green_kernel, red_kernel):
    """
    demosaicing 3 channels then return

    :return: demosaic_blue_channel, demosaic_green_channel, demosaic_red_channel
    """
    # 2D convolution
    demosaic_blue_channel = cv2.filter2D(src=bayer_image, ddepth=-1, kernel=blue_kernel)
    demosaic_green_channel = cv2.filter2D(src=bayer_image, ddepth=-1, kernel=green_kernel)
    demosaic_red_channel = cv2.filter2D(src=bayer_image, ddepth=-1, kernel=red_channel)

    return demosaic_blue_channel, demosaic_green_channel, demosaic_red_channel


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
            bayer,
            blue_channel, green_channel, red_channel,
            blue_kernel, green_kernel, red_kernel)

    print(demosaic_blue_channel)
    print(demosaic_green_channel)
    print(demosaic_red_channel)



if __name__ == '__main__':
    main()
