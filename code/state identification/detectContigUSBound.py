import numpy as np
import cv2 as cv
import sys
import os

path = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\USStateChoro\finalTest'
testImageDir = os.listdir(path)
for img in testImageDir:
# imageName = 'map_wm_persons.jpg'
    im = cv.imread(path + '\\' +img)
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    thresh = cv.adaptiveThreshold(imgray, 255, cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours.sort(key=lambda x: cv.contourArea(x),reverse=True)
    im_contour = contours[1]

    cv.drawContours(im, [im_contour], -1, (0,255,0), 3)
    cv.imshow("Display window", im)
    k = cv.waitKey(0)

usa_wireframe = cv.imread(path + '\\' +imageName)
print(usa_wireframe.shape)
cv.imshow("",usa_wireframe)
k = cv.waitKey(0)

usagray = cv.cvtColor(usa_wireframe, cv.COLOR_BGR2GRAY)
thresh = cv.adaptiveThreshold(usagray, 255, cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
usa_contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

usa_contours.sort(key=lambda x: cv.contourArea(x),reverse=True)
usa_contour = usa_contours[1]

cv.drawContours(usa_wireframe, [usa_contour], -1, (0,255,0), 3)
cv.imshow("Display window", usa_wireframe)
k = cv.waitKey(0)
