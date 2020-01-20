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
            # IndexOutOfBound of TypeError
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
    blue_channel, green_channel, red_channel = \
        np.zeros((height, width), dtype=np.float32), \
        np.zeros((height, width), dtype=np.float32), \
        np.zeros((height, width), dtype=np.float32)

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


def get_root_squared_differences(original, demosaic):
    """
    compute the root squared differences between original and demosaic over 3 color channels

    :param original: original image from JPG image (height, width, 3)
    :param demosaic: demosaic image from BMP image (height, width, 3)
    :return: summed root squared differences of two images (height, width, 1)
    """
    # diff_float64 = np.sum(np.square(original - demosaic), axis=2)
    diff_float64 = np.sqrt(np.sum(np.square(original - demosaic), axis=2))
    diff_uint8 = cv2.convertScaleAbs(diff_float64)
    print(np.max(diff_uint8))
    return diff_uint8
    # return np.squcv2.convertScaleAbs(improved_demosaic)are(original - demosaic)


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

    print('5. after combining 3 demosaic channels:')
    print('demosaic.dtype =', demosaic.dtype)   # float32

    # convert the result to 8-bit integer
    demosaic = cv2.convertScaleAbs(demosaic)

    print('6. after converting to 8-bit integer:')
    print('demosaic.dtype =', demosaic.dtype)   # uint8

    # compute the root squared differences between the original and the demosaic
    root_squared_diff = get_root_squared_differences(original, demosaic)
    print('root_squared_diff.dtype =', root_squared_diff.dtype)     # uint8

    # important for presentation!!
    # convert to unsigned int8 type so that the cv2.imshow will show using
    # range [0, 255] instead of [0, 1]

    ###### present the concatenated images using plt.imshow() #####
    # we can change the plot size to make the images bigger
    plt.figure(figsize=(16, 10))

    # 1st subplot (original)
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.xticks([])
    plt.yticks([])

    # 2nd subplot (demosaic)
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(demosaic, cv2.COLOR_BGR2RGB))
    plt.title('Demosaic Image')
    plt.xticks([])
    plt.yticks([])

    # 3rd subplot (squared differences)
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(root_squared_diff, cv2.IMREAD_GRAYSCALE))
    plt.title('Root Squared Differences')
    plt.xticks([])
    plt.yticks([])

    plt.show()

    ###### present the concatenated images using cv2.imshow #####
    # concatenate the 3 images horizontally (side by side)
#     concat = np.concatenate((original, demosaic, squared_diff), axis=1)
#
#     cv2.imshow(winname='Part 1 output', mat=concat)


def part_two_show(original, demosaic_blue_channel, demosaic_green_channel, demosaic_red_channel):
    """
    apply the improved interpolation approach

    :param original:                JPG image (shape of (height, width, no. of channels))
    :param demosaic_blue_channel:   demosaic blue channel
    :param demosaic_green_channel:  demosaic green channel
    :param demosaic_red_channel:    demosaic red channel
    """
    ###### simple bilinear interpolation approach #####
    # demosaic_red_channel does not change

    # important for cv2.medianBlur!!
    # input array passed to cv2.medianBlur must be converted to int8 or
    # float32

    # computing the difference images G-R and B-R
    diff_green_red = demosaic_green_channel - demosaic_red_channel
    diff_blue_red = demosaic_blue_channel - demosaic_red_channel

    # applying median filtering to the images G-R and B-R
    median_green_red = cv2.medianBlur(src=diff_green_red, ksize=3)
    median_blue_red = cv2.medianBlur(src=diff_blue_red, ksize=3)

    # modify the G and B channels by adding the R channel to the respective
    # difference images
    modified_green = demosaic_red_channel + median_green_red
    modified_blue = demosaic_red_channel + median_blue_red

    # stack new B/G/R channels together
    improved_demosaic = np.dstack((modified_blue, modified_green, demosaic_red_channel))
    print('improved_demosaic.dtype =', improved_demosaic.dtype)     # float32

    # convert the result to 8-bit integer
    improved_demosaic = cv2.convertScaleAbs(improved_demosaic)
    print('improved_demosaic.dtype =', improved_demosaic.dtype)     # uint8

    # compute the squared differences between the original and the improved_demosaic
    root_squared_diff = get_root_squared_differences(original, improved_demosaic)
    print('root_squared_diff.dtype =', root_squared_diff.dtype)     # uint8

    # important for presentation!!
    # convert to unsigned int8 type so that the cv2.imshow will show using
    # range [0, 255] instead of [0, 1]
#     print(squared_diff.dtype)  # uint8

    ###### present the concatenated images using plt.imshow() #####
    # we can change the plot size to make the images bigger
    plt.figure(figsize=(16, 10))

    # 1st subplot (original)
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.xticks([])
    plt.yticks([])

    # 2nd subplot (improved demosaic)
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(improved_demosaic, cv2.COLOR_BGR2RGB))
    plt.title('Improved Demosaic Image')
    plt.xticks([])
    plt.yticks([])

    # 3rd subplot (squared differences)
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(root_squared_diff, cv2.IMREAD_GRAYSCALE))
    plt.title('Root Squared Differences')
    plt.xticks([])
    plt.yticks([])

    plt.show()

    ###### present the concatenated images using cv2.imshow #####
    # concatenate the 3 images horizontally (side by side)
#     concat = np.concatenate((original, improved_demosaic, squared_diff), axis=1)
#
#     cv2.imshow(winname='Part 2 output', mat=concat)


def main():
    """
    main function
    """
    # read the images
    original, bayer = read_images()

    print('1. after reading images:')
    print('original.dtype =', original.dtype)
    print('bayer.dtype =', bayer.dtype)

    # separate channels for bayer image
    blue_channel, green_channel, red_channel = separate_channels(bayer)

    print('2. after separating channels:')
    print('original.dtype =', original.dtype)
    print('bayer.dtype =', bayer.dtype)
    print('blue_channel.dtype =', blue_channel.dtype)
    print('green_channel.dtype =', green_channel.dtype)
    print('red_channel.dtype =', red_channel.dtype)

    # initialize kernel for each channel
    blue_kernel, green_kernel, red_kernel = get_channel_kernels()

    print('3. after initializing the 3 kernels:')
    print('blue_kernel.dtype =', blue_kernel.dtype)
    print('green_kernel.dtype =', blue_channel.dtype)
    print('red_kernel.dtype =', red_kernel.dtype)

    # demosaicing
    demosaic_blue_channel, demosaic_green_channel, demosaic_red_channel = \
        get_demosaic_channels(blue_channel, green_channel, red_channel,
                                blue_kernel, green_kernel, red_kernel)

    print('4. after applying cv2.filter2D:')
    print('demosaic_blue_channel.dtype =', demosaic_blue_channel.dtype)
    print('demosaic_green_channel.dtype =', demosaic_green_channel.dtype)
    print('demosaic_red_channel.dtype =', demosaic_red_channel.dtype)

    # part 1: show the concatenated 3 output images
    part_one_show(original, demosaic_blue_channel, demosaic_green_channel,
        demosaic_red_channel)

    # part 2: apply the improved approach then show
    part_two_show(original, demosaic_blue_channel, demosaic_green_channel,
        demosaic_red_channel)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()

