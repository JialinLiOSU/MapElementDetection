# this file is used to crop the legend areas from labeled images
# the cropped legend image will be analyzed further
# Author: Jialin Li
# Date: 4/20/2020

import cv2
import matplotlib

dataPath = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Name Entity Recognition'
img = cv2.imread("lenna.png")
crop_img = img[y:y+h, x:x+w]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)
