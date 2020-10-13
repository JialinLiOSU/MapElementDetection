# this file is used to put functions together to analyze legend of choropleth map
# Author: Jialin Li
# Date: 6/3/2020
import os, io
import pandas as pd
import cv2
import random
from PIL import Image, ImageDraw
from enum import Enum
import easyocr
import numpy as np
import pickle

def extractNumbers(str1):
    textBlocks = str1.split(' ')
    numbers = []
    for text in textBlocks:
        if text.isNumeric():
            numbers.append(text)
    return numbers

def detectRects(image_file):
    img = cv2.imread(image_file)
    height = img.shape[0]
    width = img.shape[1]
    enlargeRatio = 3
    dim = (width*3, height*3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray,(3,3),0)
    edge = cv2.Canny(gray, 50,200) 
    # Taking a matrix of size 5 as the kernel 
    kernel = np.ones((3,3), np.uint8) 

    n= 2
    for i in range(n):
        edge = cv2.dilate(edge, kernel, iterations=1) 
        edge = cv2.erode(edge, kernel, iterations=1) 

    (contours, _) = cv2.findContours(
        edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
                cv2.putText(img, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0))
                rectList.append(approx)
    return rectList

def legendSymbolAlign(rectList):
    # based on a list of bbox of texts, to identify the alignment of legend symbols
    # output can be 0,1,2 representing horizontal, vertical and 2 dimensional
    

    numRect = len(rectList)
    if numRect == 1:
        return 2

    # identify whether there are intersections of the whole column or row for each rect

def main():
    dataPath = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Legend Analysis\\enhanced legend images'
    image_name = 'ChoImg101_0.jpg'
    image_file = dataPath + '\\' + image_name
    
    # detect rectangles 
    rectList = detectRects(image_file) 
    # detect texts
    reader = easyocr.Reader(['en'])
    bounds = reader.readtext(image_file)

    with io.open(image_file, 'rb') as image_file1:
        content = image_file1.read()
    content_image = types.Image(content=content)
    response = client.document_text_detection(image=content_image)
    document = response.full_text_annotation
    text1 = document.text
    textByLine = text1.split('\n')
    numberByLineList = []
    for text in textByLine:
        numbers = extractNumbers(text)
        numberByLineList.append(numbers)
    text2 = text_within(document, 0,0,145,242)
    print (text1)

    numRect = len(rectList)
    numCategories = len(numberByLineList)

if __name__ == "__main__":
    main()

