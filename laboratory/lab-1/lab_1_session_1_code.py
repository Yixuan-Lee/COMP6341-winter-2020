import cv2 as cv
import numpy as np

#  --- Loading images and basics ---

img = cv.imread("sample.jpg")
cv.imshow("RGB Image", img)
cv.waitKey(0)
cv.destroyAllWindows() # Waitkey function stops the program from going forward, destroy all window closes the image as well once you press a button. Without this all of the windows will be at the monitor. So its better to use this after waitkey.
print(img.shape)
print(img.dtype)

# --- Splitting channels

# b_img, g_img, r_img each now has one channel each from the original image
# you can also use b_img = img[:, :, 0] (Use 1 and 2 for Green and Red)
b_img, g_img, r_img = cv.split(img)


# We can recreate the original image using these channels using merge

recreated_image = cv.merge((b_img, g_img, r_img))
cv.imshow("Recreated image", recreated_image)
cv.waitKey(0)
cv.destroyAllWindows()

# --- Grayscale ---

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Grayscale", img_gray)
cv.waitKey(0)
cv.destroyAllWindows()
print(img_gray.shape)
print(img_gray.dtype)


# Access the RGB pixel located at x=50, y=100, Keep in mind that
# OpenCV stores images in BGR order rather than RGB

b_pixel, g_pixel, r_pixel = img[100, 50]
print("B={}, G={}, R={}".format(b_pixel, g_pixel, r_pixel))

# Extract ROI / ROI = Region of interest, its like a patch in the image
roi = img[50:180, 20:300]
cv.imshow("roi", roi)
cv.destroyAllWindows()
cv.waitKey(0)

# Numpy is a optimized library for fast array calculations.
# So simply accessing each and every pixel values and modifying it will be very slow and it is discouraged.

# accessing RED value
print(img.item(10,10,2))

# modifying RED value / better way of accessing
img.itemset((10,10,2),100)
print(img.item(10,10,2))

# Example of clipping due to overflow

x = np.uint8([250])
y = np.uint8([10])
print(cv.add(x,y)) # 250+10 = 260 => 255
print(x+y)          # 250+10 = 260 % 256 = 4

# Filters in OpenCV
# The kernel is the window you slide across your image. The filter2d function
# does the job for you. You dont need separate for loops to do this for you.

kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]], np.float32)

img_mod = cv.filter2D(img, -1, kernel) # Here the 2nd argument is for image depth, -1  means destination image has depth same as input
cv.imshow("Sharpened", img_mod)
cv.waitKey(0)
cv.destroyAllWindows()

# Try the box filter? Or blurring?

kernel_2 = np.array([[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]], np.float32)

box_blurred = cv.filter2D(img, -1, kernel_2/9)
cv.imshow("blurred", box_blurred)
cv.waitKey(0)
cv.destroyAllWindows()

gaussian_blurred = cv.GaussianBlur(img, (3,3), 0)
cv.imshow("Blurred", gaussian_blurred)
cv.waitKey(0)
cv.destroyAllWindows()

# To see the both image side by side, you can use numpy, there are other ways too. Hstack means horizontal stack. You can use vstack. Which will position the images one under the other
blurry_merged = np.hstack((box_blurred, gaussian_blurred))
cv.imshow("Side by side image", blurry_merged)
cv.waitKey(0)
cv.destroyAllWindows()

# --- TASK ---

blank_img = np.zeros((100, 100))
cv.imshow("Blank image", blank_img)
cv.waitKey(0)
blank_img.itemset((50, 50), 255)
cv.imshow("Blank image 2", blank_img)
cv.waitKey(0)
cv.destroyAllWindows()

box_filter = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]], np.uint8)
blank_img_blurred = cv.filter2D(blank_img, -1, box_filter/9)
cv.imshow("Blurred", blank_img)
cv.waitKey(0)
cv.destroyAllWindows()

cv.GaussianBlur(blank_img, (3,3), 0, blank_img)
cv.imshow("Blurred", blank_img)
cv.waitKey(0)
cv.destroyAllWindows()
