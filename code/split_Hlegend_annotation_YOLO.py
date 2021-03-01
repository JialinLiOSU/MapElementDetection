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

path = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\generated images for detection'

# read annotation file
annoFilename='generated_Hlegend_annotation_WO_YOLO.txt'
file = open(path +'\\'+ annoFilename,'r')
annotations = file.read()
# file.writelines(incorrectImgNameStrList)
file.close() 
lineList = annotations.split('\n')
lineList = lineList[0:-1] # remove the last empty line

# count = 0
for line in lineList:
    elements = line.split(',')
    fileName = elements[0]
    
    strTemp = elements[1] + ',' + elements[2]  + ',' + elements[3] + ',' + elements[4] + ',' + elements[5] + '\n'

    # save to 
    filename=fileName.split('.')[0] + '.txt'
    file = open(path + '\\' + filename,'a')
    file.write(strTemp)
    # file.writelines(incorrectImgNameStrList)
    file.close() 

    # cv2.rectangle(img,startPoint,endPoint,(255, 0, 0),2)
    # cv2.imshow(fileName, img)
    # cv2.imwrite(path + fileName, img) 
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


