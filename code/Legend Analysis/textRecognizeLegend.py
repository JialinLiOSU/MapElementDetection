# this file is used to recognize the texts in enhanced legend area
# and identify number ranges for each category
# Author: Jialin Li
# Date: 5/4/2020
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

def main():
    dataPath = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Legend Analysis\\enhanced legend images'
    image_name = 'ChoImg101_0.jpg'
    image_file = dataPath + '\\' + image_name
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\\Users\\jiali\\Desktop\\MapElementDetection\\dataCollection\\VisionAPIDemo\\ServiceAccountToken.json'
    
    image  = Image.open(image_file)
    client = vision.ImageAnnotatorClient()
    with io.open(image_file, 'rb') as image_file1:
        content = image_file1.read()
    content_image = types.Image(content=content)
    response = client.document_text_detection(image=content_image)
    document = response.full_text_annotation
    
    
    text = text_within(document, 0,0,145,242)
    print (text)
    # print(document[0])

# def main():
#     dataPath = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Legend Analysis\\enhanced legend images'
#     os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\\Users\\jiali\\Desktop\\MapElementDetection\\dataCollection\\VisionAPIDemo\\ServiceAccountToken.json'
#     client = vision.ImageAnnotatorClient()
#     for filename in os.listdir(dataPath):
#         # read 
#         img = cv2.imread(dataPath + "\\"+filename)
#         cv2.imshow("shapes", img)

#         with io.open(os.path.join(dataPath, filename), 'rb') as image_file:
#             content = image_file.read()

#         image = vision.types.Image(content=content)

#         response = client.text_detection(image= image)
#         df = pd.DataFrame(columns=['locale','description'])
#         desc = response.text_annotations[0]
#         test1 = desc['description']
#         print(desc)
#         break

if __name__ == "__main__":
    main()

