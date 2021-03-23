# Match legend symbols and legend texts

import pickle
import os
# import easyocr
# reader = easyocr.Reader(['en']) # set OCR for English recognition
from shapely.geometry import Polygon
from shapely.geometry import box
import cv2
import numpy as np
import matplotlib.pyplot as plt  

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def isValueText(inputString):
    if not hasNumbers(inputString):
        return False
    inputString = inputString.replace(" ", "") # remove space in the string
    countDigit = sum(c.isdigit() for c in inputString) # count number of digit in string
    countDot = sum(c=='.' for c in inputString)
    countNumber = countDigit + countDot
    if countNumber/len(inputString) > 0.5:
        return True
    else:
        return False

def alignmentValueText(legendTextBboxes):
    #####  identify the alignment of (symbol + text), haven't consider 2D legend  #####
    # get all legend text bboxes with numbers
    align = -1      # 0,1,2 representing vertical, horizontal and 2 dimensional; -1 means no numeric legend
    numTextBboxes = []
    for textbbox in legendTextBboxes:
        if isValueText(textbbox[1]):
            numTextBboxes.append(textbbox)
    # identify alignment: if the medium diff of minX of numberic text bbox are smaller than width of a text bbox
    numTextShapelyBoxList = []
    numTextBboxWidthList = []
    numTextBboxHeightList = []
    if len(numTextBboxes) == 0:
        return align,numTextShapelyBoxList,numTextBboxes,numTextBboxHeightList

    for numbound in numTextBboxes:
        numTextShapelyBox = box(numbound[0][0][0],numbound[0][0][1],numbound[0][2][0],numbound[0][2][1])
        width = numbound[0][2][0] - numbound[0][0][0]
        height = numbound[0][2][1] - numbound[0][0][1]
        numTextShapelyBoxList.append(numTextShapelyBox)
        numTextBboxWidthList.append(width)
        numTextBboxHeightList.append(height)

    minWidthNumTextBbox = min(numTextBboxWidthList)

    deltaMinXList = []
    if len(numTextShapelyBoxList) == 1:
        align = 0 # vertical
        return align,numTextShapelyBoxList,numTextBboxes,numTextBboxHeightList
    for i in range(1,len(numTextShapelyBoxList)):
        minXFormer = numTextShapelyBoxList[i-1].bounds[0]
        minXLatter = numTextShapelyBoxList[i].bounds[0]
        deltaMinX = abs(minXLatter - minXFormer)
        deltaMinXList.append(deltaMinX) # length should be len(numTextBboxes) - 1
    deltaMinXList.sort()
    mediDeltaMinX = deltaMinXList[int(len(deltaMinXList) / 2)]
    if mediDeltaMinX < minWidthNumTextBbox:
        align = 0 # vertical
    else:
        align = 1 # horizontal
    return align,numTextShapelyBoxList,numTextBboxes,numTextBboxHeightList

def most_common(lst):
    return max(set(lst), key=lst.count)

def findVertRangeText(legendTextShapelyBoxList):
    # get most common text height
    textHeightList = []
    for legendTextShapelyBox in legendTextShapelyBoxList:
        textHeight = legendTextShapelyBox.bounds[3] - legendTextShapelyBox.bounds[1]
        textHeightList.append(textHeight)
    mostCommonTextHeight = most_common(textHeightList)
    numText = len(legendTextShapelyBoxList)

    return mostCommonTextHeight

def isVertIntersectBoxBox(legendTextShapelyBox,vertIntervalText):
    isVertIntersect = False
    yMin = legendTextShapelyBox.bounds[1]
    yMax = legendTextShapelyBox.bounds[3]
    yMinCur = vertIntervalText[0]
    yMaxCur = vertIntervalText[1]
    if (yMin >= yMinCur and yMin <= yMaxCur) or  (yMin < yMinCur and yMax > yMinCur):
        isVertIntersect = True
    return isVertIntersect

def isVertIntersectBoxList(legendTextShapelyBox,vertIntervalTextList):
    isVertIntersect = False
    yMin = legendTextShapelyBox.bounds[1]
    yMax = legendTextShapelyBox.bounds[3]
    indexVertIntervalTextList = []
    for i in range(len(vertIntervalTextList)):
        yMinCur = vertIntervalTextList[i][0]
        yMaxCur = vertIntervalTextList[i][1]
        if (yMin >= yMinCur and yMin <= yMaxCur) or  (yMin < yMinCur and yMax > yMinCur):
            isVertIntersect = True
            indexVertIntervalTextList.append(i)
    return isVertIntersect
    
def findVertIntervalTextList(legendTextShapelyBoxList,legendTextBboxes):
    vertIntervalTextList = []
    textInLines = []
    for i in range(len(legendTextShapelyBoxList)):
        if not isVertIntersectBoxList(legendTextShapelyBoxList[i],vertIntervalTextList):
            vertIntervalTextList.append((legendTextShapelyBoxList[i].bounds[1],legendTextShapelyBoxList[i].bounds[3]))
    return vertIntervalTextList

# find the dominant color value for each legend rectangle
def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]

def getTextsForLegendRect(LegendRectBounds,legendTextShapelyBoxList,legendTextBboxes):
    vertRangeLegendRect = (LegendRectBounds[1],LegendRectBounds[3])
    text = ''
    for i in range(len(legendTextShapelyBoxList)):
        if isVertIntersectBoxBox(legendTextShapelyBoxList[i],vertRangeLegendRect):
            text = text + ' ' + legendTextBboxes[i][1]
    return text

def main():
    # read detection results from pickle file
    legendResultsName = r'C:\Users\jiali\Desktop\MapElementDetection\code\postProcessingDetection\legendFinalBadResults.pickle'
    testImagePath = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\USStateChoro\finalTestBad'
    testImageDir = os.listdir(testImagePath)
    with open(legendResultsName, 'rb') as flegendResultsResults:
        # legend results format: (imgName,finalLegendBox,legendRectShapeBoxList,legendTextShapelyBoxList,legendTextBboxes))
        legendResults = pickle.load(flegendResultsResults)

    legendMatchingResults = []
    for legResult in legendResults:
        imgName = legResult[0]
        if imgName == '93.kisspng-united-states-choropleth-map-u-s-state-botched-5b1f544c3c1997.3561436615287798522462.jpg' \
            or imgName == 'PerCapitaMap-LN-jumbo.png':
            continue
        print(imgName)
        img = cv2.imread(testImagePath + '\\'+imgName)
        finalLegendBox = legResult[1]
        legendRectShapeBoxList = legResult[2]
        legendTextShapelyBoxList =legResult[3]
        legendTextBboxes = legResult[4]

        numRectBoxes = len(legendRectShapeBoxList)
        numTextBoxes = len(legendTextBboxes)

        if numRectBoxes == 0:
            print("No legend symbol to match")
            continue
        if numTextBoxes == 0:
            print("No text to match")
            continue

        vertRangeText = findVertRangeText(legendTextShapelyBoxList)

        vertIntervalTextList = findVertIntervalTextList(legendTextShapelyBoxList,legendTextBboxes)

        procLegendRectBoundsList = []
        for legendRectShapeBox in legendRectShapeBoxList:
            rectHeight = legendRectShapeBox.bounds[3] - legendRectShapeBox.bounds[1]
            if rectHeight >= vertRangeText * 2:
                for vertIntervalText in vertIntervalTextList:
                    if isVertIntersectBoxBox(legendRectShapeBox,vertIntervalText):
                        LegendRectBounds = (legendRectShapeBox.bounds[0],vertIntervalText[0],\
                            legendRectShapeBox.bounds[2],vertIntervalText[1])
                        procLegendRectBoundsList.append(LegendRectBounds)
                    # print('test')
            else:
                LegendRectBounds = legendRectShapeBox.bounds
                procLegendRectBoundsList.append(LegendRectBounds)

        textList = []
        dominantColorList = []
        for LegendRectBounds in procLegendRectBoundsList:
            # from LegendRectBounds to get corresponding texts
            texts = getTextsForLegendRect(LegendRectBounds,legendTextShapelyBoxList,legendTextBboxes)
            # legendRectShapeBox
            startPoint = (int(LegendRectBounds[0]), int(LegendRectBounds[1]))
            endPoint = (int(LegendRectBounds[2]), int(LegendRectBounds[3]))
            # get pixel array for current legend rect
            crop_img = img[startPoint[1]:endPoint[1], startPoint[0]:endPoint[0]]
            dominantColor = unique_count_app(crop_img)
            dominantColorList.append(dominantColor)
            textList.append(texts)
        legendMatchingResults.append((imgName, dominantColorList,textList))
        # print('test')

    with open(r'C:\Users\jiali\Desktop\MapElementDetection\code\Legend Analysis\legendMatchingResultsFinalBad.pickle', 'wb') as f:
	    pickle.dump(legendMatchingResults,f)

    

if __name__ == "__main__":    main()

