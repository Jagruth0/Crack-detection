import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("/Users/asrithkothapalli/Desktop/B.Tech Project/untitled folder/Copy of PXL_20230329_100058589.jpg",0)
img1= cv2.resize(image, (500,500), interpolation=cv2.INTER_CUBIC)
# cv2.imshow("check",img1)
blur1 = cv2.medianBlur(img1, 3)
blur = cv2.bilateralFilter(blur1,5,75,75)

blur2 = cv2.bilateralFilter(blur,5,75,75)
img=blur2
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
cv2.imwrite("skeleton.png",skel)
# cv2.imshow("dilate-skeleton",cv2.dilate(skel, element))

# dist = cv2.distanceTransform(img, cv2.DIST_L2, 3)
# _,mv,_,mp = cv2.minMaxLoc(dist)
# print(mv*2, mp) # (half)width*2, pos
# draw = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# cv2.line(draw, (0,mp[1]), (img.shape[1],mp[1]), (0,0,200), 3, -1)
# cv2.imshow("dist", draw)
# cv2.waitKey()

plt.figure()


plt.imshow(skel)
plt.show()

cv2.waitKey(0)
# import numpy as np 
# import cv2

# inputImage = cv2.imread("on.jpg")
# inputImageGray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(inputImageGray,150,200,apertureSize = 3)
# minLineLength = 30
# maxLineGap = 5
# lines = cv2.HoughLinesP(edges,cv2.HOUGH_PROBABILISTIC, np.pi/180, 30, minLineLength,maxLineGap)
# for x in range(0, len(lines)):
#     for x1,y1,x2,y2 in lines[x]:
#         #cv2.line(inputImage,(x1,y1),(x2,y2),(0,128,0),2, cv2.LINE_AA)
#         pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
#         cv2.polylines(inputImage, [pts], True, (0,255,0))

# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(inputImage,"Tracks Detected", (500, 250), font, 0.5, 255)
# cv2.imshow("Trolley_Problem_Result", inputImage)
# cv2.imshow('edge', edges)


