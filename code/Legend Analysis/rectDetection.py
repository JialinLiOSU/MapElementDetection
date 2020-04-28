# this file aims at extracting the color value of different categories in legend
# Author: Jialin Li
# Date: 4/27/2020

import cv2
import numpy as np
font = cv2.FONT_HERSHEY_COMPLEX

legendPath = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Legend Analysis\\legend images'
img = cv2.imread(legendPath + "\\"+"ChoImg214_0.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
# y,x = threshold.shape
# for row in range(y):
#     for col in range(x):
#         if (threshold[row,col] == 255):
#             threshold[row,col] = 0
#         else:
#             threshold[row,col] = 255


contours, _ = cv2.findContours(
    threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.1*cv2.arcLength(cnt, True), True)
    cv2.drawContours(img, [approx], 0, (120, 120, 120), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    
    if 4<=len(approx) <= 6:
        
        test1 = approx[0][0][1]
        test2 = approx[2][0][1]
        if abs (test1 - test2 ) > 20:
            cv2.putText(img, "Rectangle", (x, y), font, 1, (0))
            print('Rectangle')
            print(approx)
            print('/n')
    # if 6 < len(approx) < 15:
    #     cv2.putText(img, "Ellipse", (x, y), font, 1, (0))
    #     print('Ellipse')
    #     print(approx)
    #     print('/n')
    # else:
    #     # cv2.putText(img, "Circle", (x, y), font, 1, (0))
    #     if approx.ravel()[1]-approx.ravel()[3] > 10:
    #         print('Rectangle')
    #         print(approx)
    #     # print('Circle')
    #     # # print(approx)
    #     print('/n')

cv2.imshow("shapes", img)
cv2.imshow("Threshold", threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()
