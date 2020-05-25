# Assignment 2

## 1. Implemented features:

In the assignment 2, the implemented features are:

    1. Feature detection
        1.1 computation of Harris matrix
        1.2 computation of response value (corner strength)
        1.3 identify interest points
        1.4 non-maximum suppression
        1.5 adaptive non-maximum suppression (used by default)
    
    2. Feature descriptor
        2.1 angle descriptor
        2.2 vote descriptor
        2.3 contrast invariant (through chopping at 0.2 and normalization)
        2.4 rotation invariant (have a few mismatches)
        
    3. Feature matching
        3.1 match corresponding feature descriptors
        3.2 SSD distance
        3.3 ratio test

    (I mark each part as a line of comment in code, such as 
    ```############## 1.1 computation of Harris matrix ##############```)
    

## 2. Input Examples:

In the table below, I provide a few tuned input (in each row), you can use 
the examples provided here as a general test.

|  Testing Purpose  |      image 1 path       |      image 2 path       | Harris Corner threshold |      ssd_threshold      |      ratio_test      |
| ----------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- | -------------------- |
| Translation       | yosemite/Yosemite1.jpg  |  yosemite/Yosemite2.jpg |            90           |           700           |          0.7         |
| Rotation          | yosemite/Yosemite2.jpg  |  yosemite/Yosemite2rot.jpg |        100           |          2000           |          0.9        |
| Repetitive texture| panorama/pano1_0008.jpg | panorama/pano1_0009.jpg |            60           |           800           |          0.7         |
| Repetitive texture| panorama/pano1_0010.jpg | panorama/pano1_0011.jpg |            50           |           900           |          0.7         |
| Rotation          | graf/img2.ppm           | graf/img4.ppm           |            60           |          1200           |          0.7         |
|                   |                         |                         |                         |                         |                      |              


## 3. Comments:

Due to the implementation of adaptive non-maximum suppression in 
```feature_detection.py```, the running time might be long (0~3 minutes)
for large images.

So please don't enter really low Harris Corner threshold.

No extra in bonus part (except for the mandatory 1st and the 2nd extras) 
is implemented in code.


# References

1. [Harris Corner Detector in OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html)

2. [cv2.cv.cornerHarris (docs)](https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=cornerharris#cornerharris)

3. [cv2.Scharr for computing derivative (docs)](https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=scharr#scharr)

4. [cv2.GaussianBlur (docs)](https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#gaussianblur)

5. [Normalizing images in OpenCV](https://stackoverflow.com/questions/38025838/normalizing-images-in-opencv/38041997)

6. [cv2.KeyPoint (docs)](https://docs.opencv.org/3.4/d2/d29/classcv_1_1KeyPoint.html#a2990cc1848eeb189cac9709f04c8f4d3)

7. [math library in python](https://docs.python.org/3.0/library/math.html)

8. [normalize an array so that the sum of square of elements = 1](https://scicomp.stackexchange.com/questions/22094/normalize-data-so-that-the-sum-of-squares-1)

9. [cv.drawMatches (docs)](https://docs.opencv.org/master/d4/d5d/group__features2d__draw.html#gad8f463ccaf0dc6f61083abd8717c261a)

10. [cv.DMatch (docs)](https://docs.opencv.org/3.4/d4/de0/classcv_1_1DMatch.html)

11. [rotational invariance in sift descriptor (in section 5)](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)

12. [SIFT medium blog 1](https://medium.com/@lerner98/implementing-sift-in-python-36c619df7945)

13. [Are numpy arrays passed by reference?](https://stackoverflow.com/questions/11585793/are-numpy-arrays-passed-by-reference)

14. [Descriptor normalization](https://github.com/rmislam/PythonSIFT)

