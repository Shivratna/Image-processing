from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

def midpoint(A,B):
    return((A[0]+B[0])*0.5,(A[1]+B[1])*0.5)

# Gray and blur it slightly for further process

image = cv2.imread("example_01.png")
width = 2
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
pixelsPerMetric = None

cv2.imshow("Edged",edged)
orig = image
for c in cnts:
    
    if cv2.contourArea(c)<100:
        continue
    
    orig = image.copy()
    box = cv2.minAreaRect(c)
    box = cv2.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box,dtype=int)
    
    box = perspective.order_points(box)
    cv2.drawContours(orig,[box.astype("int")],-1,(0,255,0),2)
    
    for(x,y) in box:
        cv2.circle(orig,(int(x),int(y)),5,(0,0,255),-1)
    
    (tl,tr,br,bl) = box
    (tltrX,tltrY) = midpoint(tl,tr)
    (blbrX,blbrY) = midpoint(bl,br)
    
    (tlblX,tlblY) = midpoint(tl,bl)
    (trbrX,trbrY) = midpoint(tr,br)
    
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
	# draw lines between the midpoints
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)
    	# compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

	# if the pixels per metric has not been initialized, then
	# compute it as the ratio of pixels to supplied metric
	# (in this case, inches)
    if pixelsPerMetric is None:
        pixelsPerMetric = dB / width
     
	# compute the size of the object
    dimA = dA/pixelsPerMetric
    dimB = dB/pixelsPerMetric

    # draw the object sizes on the image
    cv2.putText(orig, "{:.1f}in".format(dimA),
            (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2)
    cv2.putText(orig, "{:.1f}in".format(dimB),
            (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2)

    # show the output image
    cv2.imshow("Image", orig)
    cv2.waitKey(0)