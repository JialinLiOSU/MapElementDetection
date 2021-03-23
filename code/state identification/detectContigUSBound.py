import numpy as np
import cv2 as cv
import sys
import os
import pickle

path = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\USStateChoro\finalTestBad'
savePath = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\USStateChoro\ContigUSBoundFinalBad'
testImageDir = os.listdir(path)
USBoundResults = []
for img in testImageDir:
    if img[-4:] == 'json':
        continue
    im = cv.imread(path + '\\' +img)
    areaImg = im.shape[0] * im.shape[1]
    print(img)

    # print(im.shape)
    # cv.imshow("",im)
    # k = cv.waitKey(0)

    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    thresh = cv.adaptiveThreshold(imgray, 255, cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours.sort(key=lambda x: cv.contourArea(x),reverse=True)
    # solve the problem of frame outline of a map
    for i in range(1, len(contours)):
        if contours[i].size > 200:
            im_contour = contours[i]
            break

    # im_contour = contours[1]

    cv.drawContours(im, [im_contour], -1, (0,255,0), 3)

    USBoundResults.append((img,im_contour,contours))
    cv.imwrite(savePath + '\\' + img, im) 
    # cv.imshow("Display window", im)
    # k = cv.waitKey(0)
with open(r'C:\Users\jiali\Desktop\MapElementDetection\code\state identification\USBoundResults.pickle', 'wb') as f:
	pickle.dump(USBoundResults,f)
    
# imageName = 'map_wm_persons.jpg'
# usa_wireframe = cv.imread(path + '\\' +imageName)
# print(usa_wireframe.shape)
# cv.imshow("",usa_wireframe)
# k = cv.waitKey(0)

# usagray = cv.cvtColor(usa_wireframe, cv.COLOR_BGR2GRAY)
# thresh = cv.adaptiveThreshold(usagray, 255, cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
# usa_contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# usa_contours.sort(key=lambda x: cv.contourArea(x),reverse=True)
# usa_contour = usa_contours[1]

# cv.drawContours(usa_wireframe, [usa_contour], -1, (0,255,0), 3)
# cv.imshow("Display window", usa_wireframe)
# k = cv.waitKey(0)
