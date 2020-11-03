# this file aims at extracting the color value of different categories in legend
# Author: Jialin Li
# Date: 4/27/2020

import cv2
import numpy as np
import pandas as pd  
# import numpy as np  
import matplotlib.pyplot as plt  
# import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

font = cv2.FONT_HERSHEY_COMPLEX

legendPath = r'C:\Users\jiali\Desktop\MapElementDetection\code\map image segmentation\sample images'
img = cv2.imread(legendPath + "\\"+"ChoImg21.jpg")
height = img.shape[0]
width = img.shape[1]
enlargeRatio = 1
dim = (width*enlargeRatio, height*enlargeRatio)
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blurred = cv2.medianBlur(gray, 5)
blurred = cv2.GaussianBlur(gray,(3,3),0)
edge = cv2.Canny(gray, 50,200)
laplacian = cv2.Laplacian(gray,cv2.CV_8UC1)
# Taking a matrix of size 5 as the kernel 
kernel = np.ones((3,3), np.uint8) 

n= 2
for i in range(n):
    laplacian = cv2.dilate(laplacian, kernel, iterations=1) 
    laplacian = cv2.erode(laplacian, kernel, iterations=1) 

# _, threshol0d = cv2.threshold(blurred, 23, 255, cv2.THRESH_BINARY)
_,threshold = cv2.threshold(gray,30,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# threshold = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)
# blurred = cv2.medianBlur(threshold, 3)
nrow, ncol = threshold.shape  # number of rows and columns

contours, _ = cv2.findContours(
    laplacian, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

rectList = []  # used to save the rectangles
rectIndList = []  # save the min max XY value for extraction

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True), True)
    cv2.drawContours(img, [approx], 0, (120, 120, 120), 1)
    x = approx.ravel()[0]
    y = approx.ravel()[1]

    if len(approx) >=3:

        test1 = approx[0][0][1]
        test2 = approx[2][0][1]
        if abs(test1 - test2) > 10:
            cv2.putText(img, "Rectangle", (x, y), font, 0.5, (0))
            rectList.append(approx)
            print('Rectangle')
            print(approx)
            print('/n')
cv2.imshow("gray", gray)
cv2.imshow("shapes", img)
cv2.imshow("edge", edge)
cv2.imshow("laplace", laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()

# minRow = 1000
# minCol = 1000
# maxRow = 0
# maxCol = 0

# for rect in rectList:
#     test1 = rect[0]
#     test2 = rect[0][0]
#     if rect[0][0][0] != 0:
#         for point in rect:
#             point = point[0]  # convert from np.array to list
#             if point[0] < minCol:
#                 minCol = point[0]
#             elif point[0] > maxCol:
#                 maxCol = point[0]
#             if point[1] < minRow:
#                 minRow = point[1]
#             elif point[1] > maxRow:
#                 maxRow = point[1]
#         print([minRow, maxRow, minCol, maxCol])
#         rectIndList.append([minRow, maxRow, minCol, maxCol])

# # get the coordinate of four vertice
# leftTop = [minRow, minCol]
# rightTop = [minRow, maxCol]
# leftBottom = [maxRow, minCol]
# rightBottom = [maxRow, maxCol]
# # get the value of approximate four vertice
# imgLeftTop = img[leftTop[0] + 5, leftTop[1] + 5, :]
# imgRightTop = img[rightTop[0] + 5, rightTop[1] - 5, :]
# imgLeftBottom = img[leftBottom[0] - 5, leftBottom[1] + 5, :]
# imgRightBottom = img[rightBottom[0] - 5, rightBottom[1] - 5, :]
# print(imgLeftTop)
# print(imgRightTop)
# print(imgLeftBottom)
# print(imgRightBottom)

# # collect data to train regression model
# coorList = []
# pixelValueList = []
# value1List = []
# value2List = []
# value3List = []
# for idxCol in range(minCol + 5, maxCol - 5, 5):
#     for idxRow in range(minRow + 5, maxRow - 5, 5):
#         coorList.append([idxRow, idxCol])
#         pixelValueList.append(hsv_img[minRow+5, idxCol, :])
#         value1List.append(hsv_img[minRow+5, idxCol, 0])
#         value2List.append(hsv_img[minRow+5, idxCol, 1])
#         value3List.append(hsv_img[minRow+5, idxCol, 2])
#     # print(img[minRow+5, idxCol, :])


# coorArray = np.array(coorList)
# pixelValueArray = np.array(pixelValueList)
# value1Array = np.array(value1List)
# value2Array = np.array(value2List)
# value3Array = np.array(value3List)

# X_train, X_test, y_train, y_test = train_test_split(coorArray, value3Array, test_size=0.2, random_state=0)

# regressor = LinearRegression()  
# regressor.fit(X_train, y_train)

# y_pred = regressor.predict(X_test)
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# cv2.imshow("shapes", img)
# cv2.imshow("Threshold", threshold)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
