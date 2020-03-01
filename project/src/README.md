# Project README

## 1. Implemented features

Step 1 ~ 4 + mandatory extra 1


## 2. Input Examples

|  Testing Purpose  |    images paths input   | Harris Corner threshold |      SSD threshold      |  ratio test threshold  | inlier threshold  | # of iterations of RANSAC |
| ----------------- | ----------------------- | ----------------------- | ----------------------- | ---------------------- | ----------------- | ------------------------- |
| Harris Corner     | Boxes.png <br/> #       |            90           |           N/A           |          N/A           |          N/A      |          N/A              |
| ALL               | Rainier1.png <br/> Rainier2.png <br/> #  |   80   |          1000           |          0.9           |          10       |          60               |
| ALL               | Rainier1.png <br/> Rainier2.png <br/> Rainier3.png <br/> # |  80  |  1000   |          0.8           |           4       |         150               | 
| ALL               | Rainier1.png <br/> Rainier2.png <br/> Rainier3.png <br> Rainier4.png <br/> # |  80  |  1000  |  0.9  |           2       |         200               | 
| ALL               | Rainier1.png <br/> Rainier2.png <br/> Rainier5.png <br/> Rainier6.png <br/> # |  75  |  1200  |  0.9 |           1       |         300               | 
| ALL               | Rainier1.png <br/> Rainier2.png <br/> Rainier3.png <br> Rainier4.png <br/> Rainier5.png <br/> Rainier6.png <br/> #|  75  | 1200 |  0.9  |  1  | 350  |
|                   |                         |                         |                         |                      |                        |                        |  

(**ALL** here means `Harris Corner + Matching + RANSAC + Stitching`)

(TODO: 6 images still not overlap perfectly, need to tune the code and paramters)


## 3. Saving images filename convention



# References

1. Assignment 2 instruction and my implementation

2. [Python integer -> character / character -> integer (stackoverflow)](https://stackoverflow.com/questions/704152/how-can-i-convert-a-character-to-a-integer-in-python-and-viceversa)

3. [cv.findHomography (doc)](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#cv.FindHomography)

4. [Singular matrix issue with Numpy (stackoverflow)](https://stackoverflow.com/questions/10326015/singular-matrix-issue-with-numpy)

5. [numpy.linalg.pinv (doc)](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.pinv.html)

6. [cv.getRectSubPix (doc)](https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#getrectsubpix)