# this file is using opencv to segment map images
# Author: Jialin Li
# Data: 6/29/2020

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
dataPath = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\dataCollection\\labeledMapsWithCategory\\images\\'
# outputPath = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Legend Analysis\\legend images\\'
imgName = 'ChoImg14.jpg'
img = cv.imread(dataPath + imgName)
cv.imshow('origional image',img)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('origional image',img)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
cv.imshow('threshed image',thresh)

# # noise removal
# kernel = np.ones((3,3),np.uint8)
# opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
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