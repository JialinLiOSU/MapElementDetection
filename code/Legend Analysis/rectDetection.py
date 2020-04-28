# this file aims at extracting the color value of different categories in legend
# Author: Jialin Li
# Date: 4/27/2020

import cv2
import numpy as np
font = cv2.FONT_HERSHEY_COMPLEX

legendPath = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Legend Analysis\\legend images'
img = cv2.imread(legendPath + "\\"+"ChoImg224_0.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
nrow, ncol = threshold.shape # number of rows and columns

contours, _ = cv2.findContours(
    threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

rectList = [] # used to save the rectangles
rectIndList = [] # save the min max XY value for extraction

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True), True)
    cv2.drawContours(img, [approx], 0, (120, 120, 120), 1)
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    
    if 4<=len(approx) <= 6:
        
        test1 = approx[0][0][1]
        test2 = approx[2][0][1]
        if abs (test1 - test2 ) > 20:
            cv2.putText(img, "Rectangle", (x, y), font, 0.5, (0))
            rectList.append(approx)
            print('Rectangle')
            print(approx)
            print('/n')

minRow = 1000
minCol = 1000
maxRow = 0
maxCol = 0

for rect in rectList:
    test1 = rect[0]
    test2 = rect[0][0]
    if rect[0][0][0] != 0:
        for point in rect:
            point = point[0] # convert from np.array to list
            if point[0] < minCol:
                minCol = point[0]
            elif point[0] > maxCol:
                maxCol = point[0]
            if point[1] < minRow:
                minRow = point[1]
            elif point[1] > maxRow:
                maxRow = point[1]
        print([minRow, maxRow, minCol, maxCol])
        rectIndList.append([minRow, maxRow, minCol, maxCol])

# get the coordinate of four vertice
leftTop = [minRow, minCol]
rightTop = [minRow, maxCol]
leftBottom = [maxRow, minCol]
rightBottom = [maxRow, maxCol]

imgLeftTop = img[leftTop[0] + 5, leftTop[1] + 5,:]
imgRightTop = img[rightTop[0] + 5, rightTop[1] - 5,:]
imgLeftBottom = img[leftBottom[0] - 5, leftBottom[1] + 5,:]
imgRightBottom = img[rightBottom[0] - 5, rightBottom[1] - 5,:]

print(imgLeftTop)
print(imgRightTop)
print(imgLeftBottom)
print(imgRightBottom)

cv2.imshow("shapes", img)
cv2.imshow("Threshold", threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()

