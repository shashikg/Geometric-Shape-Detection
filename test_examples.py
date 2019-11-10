import numpy as np
import cv2
from GSD import find_GS

debug_flag = int(input("Do you want to get output images at each process? \nEnter 1 for YES \nEnter 0 for NO \n"))
test_type = {
    0: "shapes_solid",
    1: "shapes_nonsolid",
    2: "shapes_dis",
    3: "shapes_misc",
}

tt = int(input("Enter 0 for Solid Shapes Example \nEnter 1 for Non-Solid Shapes Example \nEnter 2 for Example to distinguish between Quadrilaterals and Ellipse One \nEnter 3 for other misc examples \n"))
test_img = test_type[tt]

img = cv2.imread(test_img + ".jpg")
img = cv2.resize(img,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_CUBIC)
fimg = find_GS(img, test_img, debug_flag)

cv2.imshow("Original", img)
cv2.imshow("Final", fimg)

cv2.waitKey(0)
cv2.destroyAllWindows()
