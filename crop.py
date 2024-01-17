
from multiprocessing.connection import wait
from cv2 import imshow, imwrite
import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("/Users/asrithkothapalli/Desktop/B.Tech Project/untitled folder/Copy of PXL_20230329_100058589.jpg",0)
img1= cv2.resize(image, (500,500), interpolation=cv2.INTER_CUBIC)
# imwrite('C:\\Users\\USER\\Desktop\\s.jpg',img1)

blur1 = cv2.medianBlur(img1,1) #image to be changed to img1
blur2 = cv2.medianBlur(blur1,5)
blur = cv2.bilateralFilter(img1,9,75,75)

# imwrite('C:\\Users\\USER\\Desktop\\resized.jpg',blur)
# crop= img1[0:300, 0:170]   #r-r:c-c
imshow('cropped',img1)
# crop=img1
blur1 = cv2.medianBlur(img1, 7)
# commented on 9th may
# imwrite('F:/codes/kaggle/resized.jpg',blur1)
blur = cv2.bilateralFilter(blur1,9,75,75)
img=blur
size = np.size(img)
skel = np.zeros(img.shape,np.uint8)

ret,img = cv2.threshold(img,127,255,0)
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
done = False

img =cv2.bitwise_not(img)
original = img

while( not done):
    eroded = cv2.erode(img,element)
    temp = cv2.dilate(eroded,element)
    temp = cv2.subtract(img,temp)
    skel = cv2.bitwise_or(skel,temp)
    img = eroded.copy()

    zeros = size - cv2.countNonZero(img)
    if zeros==size:
        done = True

# cv2.imshow("original", original)
cv2.imshow("skeleton",skel)
cv2.imwrite("skek.jpg",skel)
n_white_pix = np.sum(skel == 255)
print('Number of white pixels:', n_white_pix)
cv2.waitKey(0)
