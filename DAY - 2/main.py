# Document Detection - Transformation - Extraction basics

import imutils
import cv2
from transform import perspective_transform
import numpy as np
from skimage.filters import threshold_local

image = cv2.imread('/Users/Shivratna_pvt/Desktop/Projects/IP Basics/Image-processing/DAY - 2/receipt.jpg')
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image,height=500)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
edges = cv2.Canny(blur,75,200)

cv2.imshow("Edges",edges)
cv2.waitKey(0)

cnts = cv2.findContours(edges.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse=True)[:5]

for c in cnts:
    peri = cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,0.02*peri, True)
    
    if len(approx)==4:
        screenCnt = approx
        break
    
cv2.drawContours(image,[screenCnt],-1,(0,255,0),3)
cv2.imshow("Bordered",image)

output = perspective_transform(orig, screenCnt.reshape(4, 2) * ratio)
cv2.imshow("Tranformed",output)
cv2.waitKey(0)

warped = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)
# Getting data from receipt

import pytesseract

data = pytesseract.image_to_string(output,lang="eng",config="--psm 6")
print(data)


cv2.waitKey(0)