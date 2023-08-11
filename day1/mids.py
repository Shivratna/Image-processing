import imutils
import cv2

image = cv2.imread("tasty.jpg")
cv2.imshow("Image",image)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("GRAY",gray)

edged = cv2.Canny(gray,30,150)
cv2.imshow("Edges",edged)

thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Thresholded",thresh)

thresh = cv2.GaussianBlur(thresh,(5,5),0)

cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = image.copy()

for c in cnts:
    cv2.drawContours(output,[c],-1,(240,0,159),3)
    cv2.imshow("Countours",output)

text = "Found countours: "+str(len(cnts))
cv2.putText(output,text,(10,25),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),2)
cv2.imshow("Output",output)

mask = cv2.erode(thresh.copy(),None, iterations=5)
cv2.imshow("Eroded",mask)

mask2 = cv2.dilate(thresh.copy(),None,iterations=5)
cv2.imshow("Dilated",mask2)

mask = thresh.copy()
output = cv2.bitwise_and(image,image,mask=mask)
cv2.imshow("Masked output",output)

cv2.waitKey(0)
