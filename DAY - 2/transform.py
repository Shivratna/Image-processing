import numpy as np
import cv2

def corner_points(pts):
    rect = np.zeros((4,2),dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts,axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def perspective_transform(image,pts):
    
    rect = corner_points(pts)
    (tl, tr, br, bl) = rect
    
    widthT = np.sqrt(((tl[0]-tr[0])**2)+((tl[1]-tr[1])**2))
    widthB = np.sqrt(((bl[0]-br[0])**2)+((bl[1]-br[1])**2))
    maxWidth = max(int(widthB),int(widthT))
    
    heightL = np.sqrt(((tl[0]-bl[0])**2)+((tl[1]-bl[1])**2))
    heightR = np.sqrt(((tr[0]-br[0])**2)+((tr[1]-br[1])**2))
    maxHeight = max(int(heightR),int(heightL))
    
    # Top down view in the max height and width of image
    
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]],dtype="float32")
    
    M =cv2.getPerspectiveTransform(rect,dst)
    warped = cv2.warpPerspective(image,M,(maxWidth,maxHeight))
    
    return warped
    