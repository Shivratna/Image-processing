from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

def midpoint(A,B):
    return((A[0]+B[0])*0.5,(A[1]+B[1])*0.5)

# Gray and blur it slightly for further process

image = cv2.imread("/Users/Shivratna_pvt/Desktop/Projects/IP Basics/Image-processing/day3/example_02.png")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(7,7),0)

# Perform edge detection 

edged = cv2.Canny(gray,50,100)
edged = cv2.dilate(edged,None,iterations=1)
edged = cv2.erode(edged,None,iterations=1)

# Find contours in the edge map
edged_temp = edged
cnts = cv2.findContours(edged_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Sorting contours left to right
(cnts,_) = contours.sort_contours(cnts)
ppm = None

cv2.imshow("Edged",edged)
cv2.waitKey(0)

