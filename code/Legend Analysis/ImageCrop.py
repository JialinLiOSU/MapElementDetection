# this file is used to crop the legend areas from labeled images
# the cropped legend image will be analyzed further
# Author: Jialin Li
# Date: 4/20/2020

import cv2
import matplotlib
import os
import json
import math

dataPath = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\dataCollection\\labeledMapsWithCategory\\images'

def ComputeIndexRangeLegend(jsData):
    for shape in jsData['shapes']:
        if shape['label'] == "legend":
            minX = 2000
            minY = 2000
            maxX = 0
            maxY = 0
            for point in shape['points']:
                if point[0]<minX:
                    minX = point[0]
                elif point[0]>maxX:
                    maxX = point[0]
                if point[1]<minY:
                    minY = point[1]
                elif point[1]>maxY:
                    maxY = point[1]
            minX = int(minX)
            minY = int(minY)
            maxX = math.ceil(maxX)
            maxY = math.ceil(maxY)
            return [minX, maxX, minY, maxY]
        else:
            return None

def main():
    for filename in os.listdir(dataPath):
        if filename.endswith('.jpg'):
            img = cv2.imread(filename)
            jfName = filename.split(".")[0]+'.json'
            jsonFile = open(dataPath + "\\"+jfName)
            jsData = json.load(jsonFile)
            [minX, maxX, minY, maxY] = ComputeIndexRangeLegend(jsData)
            
            crop_img = img[minY:maxY, minX:maxX]
            cv2.imshow("cropped", crop_img)
            cv2.waitKey(0)

if __name__ == "__main__":    main()



