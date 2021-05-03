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

def isHorizonIntersectBoxBox(textBox,horizonRangeLegendRect):
    isHorizonIntersect = False
    xMinText = textBox.bounds[0]
    xMaxText = textBox.bounds[2]
    xMinLegRec = horizonRangeLegendRect[0]
    xMaxLegRec = horizonRangeLegendRect[1]
    if (xMaxText >= xMinLegRec and xMaxLegRec >= xMinText) :
        isHorizonIntersect = True
    return isHorizonIntersect


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
def getTextsForLegendRectNew(LegendRectBounds,centroidLegendRect,isVertAligned,isFromSP,legendTextShapelyBoxList,legendTextBboxes):
    # dont identify whether vertically intersected
    # but find texts according to vertically or horizontally distance
    if isVertAligned:
        if not isFromSP:
            vertRangeLegendRect = (LegendRectBounds[1],LegendRectBounds[3])
            text = ''
            textBox = None
            for i in range(len(legendTextShapelyBoxList)):
                if isVertIntersectBoxBox(legendTextShapelyBoxList[i],vertRangeLegendRect):
                    text = text + ' ' + legendTextBboxes[i][1]
                    textBox = legendTextShapelyBoxList[i]
            return text,textBox
        else:
            # find text box with smallest y distance
            yTarget = centroidLegendRect.y
            yDistanceList = []
            for i in range(len(legendTextShapelyBoxList)):
                yText = legendTextShapelyBoxList[i].centroid.y
                yDistance = abs(yText - yTarget)
                yDistanceList.append(yDistance)
            indexMinYDistance = yDistanceList.index(min(yDistanceList))
            text = legendTextBboxes[indexMinYDistance][1]
            textBox = legendTextShapelyBoxList[indexMinYDistance]
    else:
        xTarget = centroidLegendRect.x
        xDistanceList = []
        vertRangeLegendRect = (LegendRectBounds[1],LegendRectBounds[3])
        for i in range(len(legendTextShapelyBoxList)):
            textBox = legendTextShapelyBoxList[i]
            isVIntersect = isVertIntersectBoxBox(textBox,vertRangeLegendRect)
            if isVIntersect:
                xText = legendTextShapelyBoxList[i].centroid.x
                xDistance = abs(xText - xTarget)
                xDistanceList.append(xDistance)
            else:
                xDistanceList.append(999999999999)
        indexMinXDistance = xDistanceList.index(min(xDistanceList))
        text = legendTextBboxes[indexMinXDistance][1]
        textBox = legendTextShapelyBoxList[indexMinXDistance]
    return text, textBox

def isVerticallyAligned(procLegendRectBoundsList):
    # identify the alignment of legend rects
    isVerticallyAligned = True
    isVertAlignedListI = []
    if len(procLegendRectBoundsList)==1:
        return isVerticallyAligned
    for i in range(len(procLegendRectBoundsList)):
        legendRectBounds = procLegendRectBoundsList[i]
        xMin = legendRectBounds[0]
        xMax = legendRectBounds[2]
        isVertAlignedListJ = []
        for j in range(len(procLegendRectBoundsList)):
            if i == j:
                isVertAlignedListJ.append(True)
                continue
            targetLegendRectBounds = procLegendRectBoundsList[j]
            xMinTarg = targetLegendRectBounds[0]
            xMaxTarg = targetLegendRectBounds[2]
            #X2 >= Y1 and Y2 >= X1
            if (xMax >= xMinTarg and xMaxTarg >= xMin):
                isVertIntersect = True
            else:
                isVertIntersect = False
            isVertAlignedListJ.append(isVertIntersect)
        if isVertAlignedListJ.count(False) > len(isVertAlignedListJ)/3:
            isVertAlignedListI.append(False)
        else:
            isVertAlignedListI.append(True)
    return isVertAlignedListI.count(True) >= len(isVertAlignedListI)*2/3

def main():
    # read detection results from pickle file
    legendResultsName = r'C:\Users\jiali\Desktop\MapElementDetection\code\Legend Analysis\legendPostProcessMappingColorResultsShuaichen.pickle'
    testImagePath = r'C:\Users\jiali\Desktop\MapElementDetection\images'
    testImageDir = os.listdir(testImagePath)
    savePath = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\USStateChoro\legendSymbolTextMatchingShuaichen'
    with open(legendResultsName, 'rb') as flegendResultsResults:
        # format: (imgName,finalLegendBox,legendRectShapeBoxList,legendRectDomiGreyList,isFromSuperPixelList,legendTextShapelyBoxList,legendTextBboxes))
        legendResults = pickle.load(flegendResultsResults)
    
    with open(r'C:\Users\jiali\Desktop\MapElementDetection\code\postProcessingDetection\titleResultsFinalGood.pickle', 'rb') as f:
	    titleResults = pickle.load(f)
    legendPostProcessMappingColorResults = []

    legendMatchingResults = []
    for i in range(len(legendResults)):
        imgName = legendResults[i][0]
        # if imgName == '93.kisspng-united-states-choropleth-map-u-s-state-botched-5b1f544c3c1997.3561436615287798522462.jpg' \
        #     or imgName == 'PerCapitaMap-LN-jumbo.png':
        #     continue
        print(imgName)
        img = cv2.imread(testImagePath + '\\'+imgName)
        finalLegendBox = legendResults[i][1]
        legendRectShapeBoxList = legendResults[i][2]
        legendRectDomiGreyList = legendResults[i][3]
        isFromSuperPixelList = legendResults[i][4]
        legendTextShapelyBoxList =legendResults[i][5]
        legendTextBboxes = legendResults[i][6]

        numRectBoxes = len(legendRectShapeBoxList)
        numTextBoxes = len(legendTextBboxes)

        if numRectBoxes == 0:
            print("No legend symbol to match")
            continue
        if numTextBoxes == 0:
            print("No text to match")
            continue

        vertRangeText = findVertRangeText(legendTextShapelyBoxList) # get most common text height

        vertIntervalTextList = findVertIntervalTextList(legendTextShapelyBoxList,legendTextBboxes)

        # if there are bboxes with a large height, break it into small ones based on texts
        procLegendRectBoundsList = []
        centroidLegendRectList = []
        dominantColorList = []
        isFromSuperPixelListMatch = []
        for i in range(len(legendRectShapeBoxList)):
            legendRectShapeBox = legendRectShapeBoxList[i]
            rectHeight = legendRectShapeBox.bounds[3] - legendRectShapeBox.bounds[1]
            isFromSuperPixel = isFromSuperPixelList[i]
            if rectHeight >= vertRangeText * 2 and not isFromSuperPixel:
                for vertIntervalText in vertIntervalTextList:
                    if isVertIntersectBoxBox(legendRectShapeBox,vertIntervalText):
                        LegendRectBox = box(legendRectShapeBox.bounds[0],vertIntervalText[0],\
                            legendRectShapeBox.bounds[2],vertIntervalText[1])
                        procLegendRectBoundsList.append(LegendRectBox.bounds)
                        centroidLegendRectList.append(LegendRectBox.centroid)
                        isFromSuperPixelListMatch.append(0)
                    # print('test')
            else:
                legendRectBounds = legendRectShapeBox.bounds
                centroidLegendRect = legendRectShapeBox.centroid
                procLegendRectBoundsList.append(legendRectBounds)
                centroidLegendRectList.append(centroidLegendRect)
                isFromSuperPixelListMatch.append(isFromSuperPixel)

        # identify alignment of the legend symbols
        isVertAligned = isVerticallyAligned(procLegendRectBoundsList)

        textList = []
        dominantColorList = []
        centroidLegendRectListVisual = []
        LegendRectBoundsListVisual = []
        textBoxList = []
        for i in range(len(procLegendRectBoundsList)):
            LegendRectBounds = procLegendRectBoundsList[i]
            centroidLegendRect = centroidLegendRectList[i]
            isFromSP = isFromSuperPixelListMatch[i]
            # from LegendRectBounds to get corresponding texts
            texts,textBox = getTextsForLegendRectNew(LegendRectBounds,centroidLegendRect,isVertAligned,isFromSP,legendTextShapelyBoxList,legendTextBboxes)
            
            # legendRectShapeBox
            startPoint = (int(LegendRectBounds[0]), int(LegendRectBounds[1]))
            endPoint = (int(LegendRectBounds[2]), int(LegendRectBounds[3]))
            # get pixel array for current legend rect
            crop_img = img[startPoint[1]:endPoint[1], startPoint[0]:endPoint[0]]
            dominantColor = unique_count_app(crop_img)
            dominantColorList.append(dominantColor)
            textList.append(texts)
            textBoxList.append(textBox)
        legendMatchingResults.append((imgName, dominantColorList,textList))
        if all([t == '' for t in textList]):
            print('imgName: '+ imgName + ', No Text matched')
            continue
        
        # visualize results
        startPoint = (int(finalLegendBox.bounds[0]), int(finalLegendBox.bounds[1]))
        endPoint = (int(finalLegendBox.bounds[2]), int(finalLegendBox.bounds[3]))
        cv2.rectangle(img,startPoint,endPoint,(255, 0, 0),2)
        for i in range(len(procLegendRectBoundsList)):
            color = list(np.random.choice(range(256), size=3))
            colorTuple = (int(color[0]), int(color[1]), int(color[2]))
            LegendRectBounds = procLegendRectBoundsList[i]
            startPoint = (int(LegendRectBounds[0]), int(LegendRectBounds[1]))
            endPoint = (int(LegendRectBounds[2]), int(LegendRectBounds[3]))
            cv2.rectangle(img,startPoint,endPoint,colorTuple,2)

            TextBox = textBoxList[i]
            if TextBox != None:
                startPoint = (int(TextBox.bounds[0]), int(TextBox.bounds[1]))
                endPoint = (int(TextBox.bounds[2]), int(TextBox.bounds[3]))
                cv2.rectangle(img,startPoint,endPoint,colorTuple,2)

        cv2.imwrite(savePath + '\\' + imgName, img) 
        # cv2.imshow(imgName, img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print('test')

    with open(r'C:\Users\jiali\Desktop\MapElementDetection\code\Legend Analysis\legendMatchingResultsFinalBad.pickle', 'wb') as f:
	    pickle.dump(legendMatchingResults,f)

    

if __name__ == "__main__":    main()

