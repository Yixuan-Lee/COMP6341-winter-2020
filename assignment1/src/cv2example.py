import cv2

imgsrc = r'../image_set/crayons.jpg'
img = cv2.imread(imgsrc, cv2.IMREAD_COLOR)
cv2.imshow('image', img)
cv2.waitKey(0)
