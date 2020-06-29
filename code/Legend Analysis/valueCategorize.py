# this file aims at extracting the color value of different categories in legend
# Author: Jialin Li
# Date: 4/21/2020

import cv2
import matplotlib
import os
import json
import math
import operator
import numpy as np

legendPath = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Legend Analysis\\legend images'

class pixelValue:
    def __init__(self,value,count=1):
        self.value = value
        self.count = count
    def __eq__(self, other):
        return self.value == other.value

def main():
    img = cv2.imread(legendPath + "\\"+"ChoImg47_0.jpg")
    valueClassList = []
    valueList = []
    for row in img:
        for pix in row:
            pv = pixelValue(pix)
            pix = list(pix)
            if pix in valueList:
                
                idx = valueList.index(pix)
                valueClassList[idx].count = valueClassList[idx].count + 1
            else:
                valueList.append(pix)
                valueClassList.append(pv)
    
    valueClassList.sort(reverse = True, key = operator.attrgetter('count'))
    for valueClass in valueClassList:
        if max(list(valueClass.value)) - min(list(valueClass.value)) > 20:
            if valueClass.count > 2:
                print (valueClass.value)
                print (valueClass.count)

    # print(img[0])

if __name__ == "__main__":    main()