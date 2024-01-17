import cv2 
import numpy as np

img = cv2.imread("/Users/asrithkothapalli/Desktop/B.Tech Project/untitled folder/Copy of PXL_20230329_100058589.jpg")
cv2.imshow('original image', img)
image= cv2.resize(img, (500,500), interpolation=cv2.INTER_CUBIC)

blur1 = cv2.medianBlur(image,11)
blur = cv2.bilateralFilter(blur1,13,75,75)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)



thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,13,3)

canny = cv2.Canny(thresh, 120, 255, 1)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
opening = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
dilate = cv2.dilate(opening, kernel, iterations=2)

cnts = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
area = 0
# print("the area is ",area)
perimeter = cv2.arcLength(cnts[0],True)
# print("perimeteer is ",perimeter)
# min_area = 4000
for c in cnts:
    area += cv2.contourArea(c)
    # perimeter = cv2.arcLength(c,True)
    # if area > min_area:
    cv2.drawContours(image, [c], -1, (36, 255, 12), 2)
print("the total area is ",area)

print("size",img.size)
from PIL import Image
pic = Image.open("/Users/asrithkothapalli/Desktop/B.Tech Project/untitled folder/Copy of PXL_20230329_100058589.jpg")
width = pic.width
height = pic.height
  
# display width and height
print("The height of the image is: ", height)
print("The width of the image is: ", width)
cv2.imshow('image', image)
cv2.imwrite('image.png', image)

# contours,_ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# cnt = contours[0]
# M = cv2.moments(cnt)
# print( M )
# area = cv2.contourArea(cnt)
# print("the area is ",area)
# perimeter = cv2.arcLength(cnt,True)
# print("perimeteer is ",perimeter)
# lst_intensities = []

# # For each list of contour points...
# for i in range(len(contours)):
#     # Create a mask image that contains the contour filled in
#     cimg = np.zeros_like(gray)
#     cv2.drawContours(cimg, contours, i, color=255, thickness=-1)

#     # Access the image pixels and create a 1D numpy array then add to list
#     pts = np.where(cimg == 255)
#     lst_intensities.append(image[pts[0], pts[1]])
    
# print (len(contours))   
# print(contourArea(contours[i])


# for i in range (len(lst_intensities)):
#     print(len(lst_intensities[i]))
# cv2.waitKey(0)



hh, ww = image.shape[:2]

# threshold on black
# Define lower and uppper limits of what we call "white-ish"
lower = np.array([0, 0, 0])
upper = np.array([0, 0, 0])

# Create mask to only select black
thresh = cv2.inRange(image, lower, upper)

# invert mask so shapes are white on black background
thresh_inv = 255-thresh  # type: ignore
img=image

big_contour = max(cnts, key=cv2.contourArea)

# draw white contour on black background as mask
mask = np.zeros((hh,ww), dtype=np.uint8)
cv2.drawContours(mask, [big_contour], 0, (255,255,255), cv2.FILLED)

# invert mask so shapes are white on black background
mask_inv = 255 - mask

# create new (blue) background
bckgnd = np.full_like(img, (255,0,0))

# apply mask to image
image_masked = cv2.bitwise_and(img, img, mask=mask)

# apply inverse mask to background
bckgnd_masked = cv2.bitwise_and(bckgnd, bckgnd, mask=mask_inv)

# add together
result = cv2.add(image_masked, bckgnd_masked)

# save results
cv2.imwrite('shapes_inverted_mask.jpg', mask_inv)
cv2.imwrite('shapes_masked.jpg', image_masked)
cv2.imwrite('shapes_bckgrnd_masked.jpg', bckgnd_masked )
cv2.imwrite('shapes_result.jpg', result)
Mask = cv2.bitwise_not(mask_inv)


dist = cv2.distanceTransform(Mask, cv2.DIST_L2, 3)
_,mv,_,mp = cv2.minMaxLoc(dist)
print('Max width is ',mv*2,'co-ordinates', mp) # (half)width*2, pos
draw = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
cv2.line(draw, (0,mp[1]), (mask_inv.shape[1],mp[1]), (0,0,200), 3, -1)
cv2.imshow("dist", draw)
cv2.imwrite("dist.png",draw)

resultimage = np.zeros((800, 800))
normalizedimage=cv2.normalize(dist, resultimage, 255, 0, cv2.NORM_MINMAX)
cv2.imshow('Normalized_image', normalizedimage)
cv2.waitKey()

# cv2.imshow('mask', mask)
# cv2.imshow('image_masked', image_masked)
# cv2.imshow('bckgrnd_masked', bckgnd_masked)
# cv2.imshow('result', result)
cv2.waitKey(0)
