import numpy as np
import cv2
from GSD import find_GS

debug_flag = int(input("Do you want to get output images at each process? \nEnter 1 for YES \nEnter 0 for NO \n"))
test_img = input("Name of your .jpg image file \n").rstrip()

img = cv2.imread(test_img + ".jpg")
fimg = find_GS(img, test_img, debug_flag)

cv2.imshow("Original", img)
cv2.imshow("Final", fimg)

cv2.waitKey(0)
cv2.destroyAllWindows()
