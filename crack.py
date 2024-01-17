import cv2 

import numpy as np
img = cv2.imread("/Users/asrithkothapalli/Desktop/B.Tech Project/untitled folder/Copy of PXL_20230329_100058589.jpg")
image= cv2.resize(img, (500,500), interpolation=cv2.INTER_CUBIC)
#image = cv2.imread("own1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)
cv2.imwrite('gtry.png',gray)
dimension=image.shape
# print('dimensions',dimension)
# def rescale(frame, scale=5):
#     width=int(frame.shape[1]*scale)
#     height=int(frame.shape[0]*scale)
#     dimensions=(width,height)
#     return cv2.cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)
# resized_image= rescale(image)
# cv2.imshow('image',resized_image)
# imag=resized_image
blur1 = cv2.medianBlur(image, 11)
cv2.imshow('median blur',blur1)
blur = cv2.bilateralFilter(blur1,11,75,75)
cv2.imshow('bilateral filter',blur)
cv2.imwrite('bilater.png',blur)
blur2 = cv2.bilateralFilter(blur,9,75,75)
cv2.imshow('bilateral filter_2',blur2)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray',gray)
thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,3)
canny = cv2.Canny(thresh, 120, 255, 1)
cv2.imshow('canny',canny)
cv2.imwrite('canny.png',canny)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
opening = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
cv2.imshow('morphology',opening)
cv2.imwrite('mor[pho.png',opening)
dilate = cv2.dilate(opening, kernel, iterations=2)
cv2.imshow('dilate',dilate)
cnts = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
area = cv2.contourArea(cnts[0])

perimeter = cv2.arcLength(cnts[0],True)

min_area = 4000
for c in cnts:
    area = cv2.contourArea(c)
    if area > min_area:
        cv2.drawContours(image, [c], -1, (36, 255, 12), 2)

cv2.imshow('image', image)
cv2.imwrite('image.png', image)

cv2.waitKey(0)


