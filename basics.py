# Show image in a window

import imutils
import cv2

image = cv2.imread('jp.png')

h,w,d = image.shape

print(f"Height: {h}, Width: {w}, Dep: {d}")

cv2.imshow("Image box",image)


res_image = cv2.resize(image,(200,200))
cv2.imshow("Resized",res_image)


# Now resize with aspect ratio

r = 300.0/w
dim = (300, int(h*r))

# res_image2 = cv2.resize(image,dim)
res_image2 = imutils.resize(image,300)

cv2.imshow("Resizedx300",res_image2)

rot1 = imutils.rotate(image,45)
rot2 = imutils.rotate_bound(image,45)

cv2.imshow("Rotate",rot1)
cv2.imshow("Rotate2",rot2)

blur = cv2.GaussianBlur(image,(11,11),0)
cv2.imshow("Blur",blur)

image_copy = image
output = cv2.putText(image_copy,"TRIAL TEXT IN IMAGE",(10,20),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),2)
cv2.imshow("Texted",output)


cv2.waitKey(0)
