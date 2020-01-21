# Assignment 1 README

The assignment 1 code is written in Python 3.6 and using PyCharm as IDE. 
The code implemented functions:

1. a linear interpolation that converts Bayer images to RGB images (Part 1)

2. a improved bilinear interpolation approach that converts Bayer image to RGB 
images (Part 2)

3. compare the root squared difference between *[Original, Demosaic (Part 1)]* 
and *[Original, Improved Demosaic (Part 2)]*


## 1. Required packages

The code requires packages:

1. Numpy

2. OpenCV

3. Matplotlib

## 2. About code
There are 3 .jpg and .bmp images hardcoded in `main.py` in variables 
`jpg_images` and `bmp_images` at the very top. If it is necessary to test 
more images, you need to add the image names to both variables correspondingly.

During testing, type the image index in `jpg_images` and `bmp_images` which 
you would like to test, the result images will show after. 

There are 2 available ways to present the result images

- using matplotlib.show()

- using cv2.show()

You can choose either way or both, and you can switch by modifying the boolean 
variables `matplot_show` and `cv2_show` at the very top. (By default, I am 
using cv2_show())


# 3. References:

1. [opencv cv::filter2D documentation](https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04)

2. [opencv cv::filter2D example 1](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html)

3. [opencv cv::filter2D example 2](https://www.programcreek.com/python/example/89373/cv2.filter2D)

4. [opencv cv2.medianBlur documentation](https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html)

5. [opencv cv2.medianBlur example 1](https://medium.com/@florestony5454/median-filtering-with-python-and-opencv-2bce390be0d1)

6. [np.dstack](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.dstack.html)

7. [Numpy Data types](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html)

8. [opencv cv::imshow doc](https://docs.opencv.org/3.0-beta/modules/highgui/doc/user_interface.html#imshow)

9. [cv2.medianBlur argument type requirement](https://stackoverflow.com/questions/48453576/opencv-error-unsupported-format-or-combination-of-formats-unsupported-combinat/48453577)

10. [Stack overflow: Difference between plt.show and cv2.imshow](https://stackoverflow.com/questions/38598118/difference-between-plt-show-and-cv2-imshow)

11. [cv2.convertScaleAbs](https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#convertscaleabs)
