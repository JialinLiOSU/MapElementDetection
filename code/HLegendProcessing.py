# change label format for generated choropleth images
# to make sure that the format is fitting the training model

import pickle
import os
# import easyocr
# reader = easyocr.Reader(['en']) # set OCR for English recognition
from shapely.geometry import Polygon
from shapely.geometry import box
import cv2
import numpy as np
import matplotlib.pyplot as plt  

path = r'C:\Users\jiali\Desktop\MapElementDetection\code\generatedHLegendImages'
imagePath = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\generatedHLegendForDetection'

# read annotation file
annoFilename='generated_Hlegend_annotation_AU.txt'
file = open(imagePath +'\\'+ annoFilename,'r')
annotations = file.read()
# file.writelines(incorrectImgNameStrList)
file.close() 
lineList = annotations.split('\n')
lineList = lineList[0:-1] # remove the last empty line

strList = []
# count = 0
for line in lineList:
    elements = line.split(',')
    fileName = elements[0]
    count = int(fileName.split('.')[0].split('_')[-1])
    img = cv2.imread(imagePath + '\\'+fileName)

    if elements[-1] == '0': # title
        width = img.shape[1]
        height = img.shape[0]
        x1 = int(float(elements[1]) * width)
        y1 = int((1 - float(elements[2]) - float(elements[4])* 1 )  * height  )
        x2 = int((float(elements[1]) + float(elements[3])) * width )
        y2 = int((1 - float(elements[2]) - float(elements[4]) * 0 ) * height )
        startPoint, endPoint = (x1, y1), (x2, y2)
    else: # legend
        width = img.shape[1]
        height = img.shape[0]
        x1 = int(float(elements[1]) * width)
        y1 = int((1 - float(elements[2]) - float(elements[4])* 1 )  * height  )
        x2 = int((float(elements[1]) + float(elements[3])) * width )
        y2 = int((1 - float(elements[2]) - float(elements[4]) * 0 ) * height  )
        startPoint, endPoint = (x1, y1), (x2, y2)
    
    # count = count + 1
    
    strTemp = fileName + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + elements[-1] + '\n'
    strList.append(strTemp)

    cv2.rectangle(img,startPoint,endPoint,(255, 0, 0),2)
    cv2.imshow(fileName, img)
    # cv2.imwrite(imagePath + fileName, img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# save to 
filename='generated_Hlegend_annotation_startend'+'.txt'
file = open(path + '\\' + filename,'a')
file.writelines(strList)
# file.writelines(incorrectImgNameStrList)
file.close() 
