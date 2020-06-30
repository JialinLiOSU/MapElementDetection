# this file is used to put functions together to analyze legend of choropleth map
# Author: Jialin Li
# Date: 6/3/2020
import os, io
from google.cloud import vision
from google.cloud.vision import types
import pandas as pd
import cv2
import random
from PIL import Image, ImageDraw
from enum import Enum

class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5
    
def draw_boxes(image, bounds, color,width=5):
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        draw.line([
            bound.vertices[0].x, bound.vertices[0].y,
            bound.vertices[1].x, bound.vertices[1].y,
            bound.vertices[2].x, bound.vertices[2].y,
            bound.vertices[3].x, bound.vertices[3].y,
            bound.vertices[0].x, bound.vertices[0].y],fill=color, width=width)
    return image

def get_document_bounds(response, feature):
    document = response.full_text_annotation
    bounds=[]
    for i,page in enumerate(document.pages):
        for block in page.blocks:
            if feature==FeatureType.BLOCK:
                bounds.append(block.bounding_box)
            for paragraph in block.paragraphs:
                if feature==FeatureType.PARA:
                    bounds.append(paragraph.bounding_box)
                for word in paragraph.words:
                    for symbol in word.symbols:
                        if (feature == FeatureType.SYMBOL):
                            bounds.append(symbol.bounding_box)
                    if (feature == FeatureType.WORD):
                        bounds.append(word.bounding_box)
    return bounds
def text_within(document,x1,y1,x2,y2): 
    text=""
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    for symbol in word.symbols:
                        min_x=min(symbol.bounding_box.vertices[0].x,symbol.bounding_box.vertices[1].x,symbol.bounding_box.vertices[2].x,symbol.bounding_box.vertices[3].x)
                        max_x=max(symbol.bounding_box.vertices[0].x,symbol.bounding_box.vertices[1].x,symbol.bounding_box.vertices[2].x,symbol.bounding_box.vertices[3].x)
                        min_y=min(symbol.bounding_box.vertices[0].y,symbol.bounding_box.vertices[1].y,symbol.bounding_box.vertices[2].y,symbol.bounding_box.vertices[3].y)
                        max_y=max(symbol.bounding_box.vertices[0].y,symbol.bounding_box.vertices[1].y,symbol.bounding_box.vertices[2].y,symbol.bounding_box.vertices[3].y)
                        if(min_x >= x1 and max_x <= x2 and min_y >= y1 and max_y <= y2):
                            text+=symbol.text
                            if(symbol.property.detected_break.type==1 or 
                               symbol.property.detected_break.type==3):
                                text+=' '
                            if(symbol.property.detected_break.type==2):
                                text+='\t'
                            if(symbol.property.detected_break.type==5):
                                text+='\n'
    return text

def extractNumbers(str1):
    textBlocks = str1.split(' ')
    numbers = []
    for text in textBlocks:
        if text.isNumeric():
            numbers.append(text)
    return numbers

def detectRects(image_file):
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    nrow, ncol = threshold.shape  # number of rows and columns

    (_, contours, _) = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rectList = []  # used to save the rectangles
    rectIndList = []  # save the min max XY value for extraction

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True), True)
        cv2.drawContours(img, [approx], 0, (120, 120, 120), 1)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        if 4 <= len(approx) <= 6:
            test1 = approx[0][0][1]
            test2 = approx[2][0][1]
            if abs(test1 - test2) > 20:
                rectList.append(approx)

    return rectList
def legendSymbolAlign(rectList):
    # based on a list of rectangles, to identify the alignment of legend symbols
    # output can be 0,1,2 representing horizontal, vertical and 2 dimensional
    numRect = len(rectList)
    if numRect == 1:
        return 2

    # identify whether there are intersections of the whole column or row for each rect
    

    


def main():
    dataPath = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Legend Analysis\\enhanced legend images'
    image_name = 'ChoImg101_0.jpg'
    image_file = dataPath + '\\' + image_name
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\\Users\\jiali\\Desktop\\MapElementDetection\\dataCollection\\VisionAPIDemo\\ServiceAccountToken.json'
    # detect rectangles 
    rectList = detectRects(image_file) 
    # detect texts
    image  = Image.open(image_file)
    client = vision.ImageAnnotatorClient()
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


    # print(document[0])



if __name__ == "__main__":
    main()

