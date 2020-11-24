# this file is using opencv to segment map images
# Author: Jialin Li
# Data: 6/29/2020

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageEnhance
dataPath = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\dataCollection\\labeledMapsWithCategory\\enhImages\\'
# outputPath = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Legend Analysis\\legend images\\'
imgName = 'ChoImg14.jpg'


img = cv.imread(dataPath + imgName)
# dst = cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(img,100,150)
# edges = cv.bitwise_not(edges)
# cv.imshow('threshed image',edges)
# cv.waitKey(0)
# ret, thresh = cv.threshold(edges,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
thresh = cv.adaptiveThreshold(edges,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)

# thresh = cv.bitwise_not(thresh)

# cv.imshow('threshed image',thresh)
# cv.waitKey(0)

# based on thresh to detect contours
nrow, ncol = thresh.shape  # number of rows and columns

(_, cnts, _)= cv.findContours(
    thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

rectList = []  # used to save the rectangles
rectIndList = []  # save the min max XY value for extraction
font = cv.FONT_HERSHEY_COMPLEX

for cnt in cnts:
    approx = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)
    
    x = approx.ravel()[0]
    y = approx.ravel()[1]

    if len(approx) >=4:
        
        test1 = approx[0][0][1]
        test2 = approx[2][0][1]
        numPoint = len(approx)
        maxX=0
        maxY=0
        minX=800
        minY=800
        for point in approx:
            x=point[0][0]
            y=point[0][1]
            if x < minX:
                minX = x
            if x > maxX:
                maxX = x
            if y < minY:
                minY = y
            if y > maxY:
                maxY = y

        if abs(maxX - minX) > 20 and abs(maxY - minY) > 20 :
            cv.drawContours(img, [cnt], 0, (10, 10, 10), 1)
            # cv.putText(img, "Rectangle", (x, y), font, 0.5, (0))
            # cv2.drawContours(img, [approx], 0, (120, 120, 120), 1)
            rectList.append(approx)
            print('Rectangle')
            print(approx)
            print('/n')

# edges = cv.Canny(thresh,100,150)


# noise removal
# kernel = np.ones((3,3),np.uint8)
# opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
cv.imshow('threshed image',img)
cv.waitKey(0)
# # sure background area
# sure_bg = cv.dilate(opening,kernel,iterations=3)
# # Finding sure foreground area
# dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
# ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# unknown = cv.subtract(sure_bg,sure_fg)

# # Marker labelling
# ret, markers = cv.connectedComponents(sure_fg)
# # Add one to all labels so that sure background is not 0, but 1
# markers = markers+1
# # Now, mark the region of unknown with zero
# markers[unknown==255] = 0

# markers = cv.watershed(img,markers)
# img[markers == -1] = [255,0,0]
# cv.imshow('threshed image',thresh)
# cv.waitKey(0)