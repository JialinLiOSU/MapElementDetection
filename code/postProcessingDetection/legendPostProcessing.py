# The code is for legend detection post processing
# General solution for titles: Do text detection using easyOCR
# if there are some overlap between text object and title object, do union and enlarge bbox
# Dilate in vertical direction by 1 * line height iteratively, until there is no text objects
import pickle
import os
# import easyocr
# reader = easyocr.Reader(['en']) # set OCR for English recognition
from shapely.geometry import Polygon
from shapely.geometry import box
import cv2
import numpy as np
import matplotlib.pyplot as plt  

# get the most likely legend bbox for imgName from detectResults
def getLegendBboxImage(imgName,detectResults):
    bboxes = []
    for result in detectResults: # result[0]: image name, result[1]: label
        if result[0] == imgName and result[1] == '1':
            bboxes.append(result)
        if result[0] != imgName and len(bboxes) > 0:
            break
    if len(bboxes) == 0:
        return None
    bbox = bboxes[0]
    for bb in bboxes:
        if bb[2]>bbox[2]:
            bbox = bb
    return bbox

def getTextBboxes(imgName, ocrResults):
    textBboxes = []
    for result in ocrResults:
        if result[0] == imgName:
            textBboxes = result[1:]
            break
    textShapelyBoxList = []
    textBboxesNew = []

    if len(textBboxes) == 0:
        return textShapelyBoxList,textBboxesNew
    for bound in textBboxes:
        if bound[2]>pow(10,-10):
            textBboxesNew.append(bound)
            textShapelyBox = box(bound[0][0][0],bound[0][0][1],bound[0][2][0],bound[0][2][1])
            textShapelyBoxList.append(textShapelyBox)
    return textShapelyBoxList,textBboxesNew

def getLegendTextShapelyBoxList(legendShapelyBox, textShapelyBoxList):
    # get text bboxes in legend area
    legendTextShapelyBoxList = [] # text bboxes in legend
    for textShapelyBox in textShapelyBoxList: # bound[0]position, bound[1] text, bound[2] probability
        if legendShapelyBox.intersects(textShapelyBox):
            legendTextShapelyBoxList.append(textShapelyBox)
    return legendTextShapelyBoxList   

def getLegendTextBboxes(legendShapelyBox, textBboxes):
    # get text bboxes in legend area
    legendTextBboxes = [] # text bboxes in legend
    for textBbox in textBboxes: # bound[0]position, bound[1] text, bound[2] probability
        textPolygon = box(textBbox[0][0][0],textBbox[0][0][1],textBbox[0][2][0],textBbox[0][2][1])
        if legendShapelyBox.intersects(textPolygon):
                legendTextBboxes.append(textBbox)
    return legendTextBboxes   

def getUnionBbox(legendShapelyBox,legendTextShapelyBoxList):
    minX = legendShapelyBox.bounds[0]
    minY = legendShapelyBox.bounds[1]
    maxX = legendShapelyBox.bounds[2]
    maxY = legendShapelyBox.bounds[3]

    for legendTextShapelyBox in legendTextShapelyBoxList:
        if (legendTextShapelyBox.bounds[0] < minX):
            minX = legendTextShapelyBox.bounds[0]
        if (legendTextShapelyBox.bounds[1] < minY):
            minY = legendTextShapelyBox.bounds[1]
        if (legendTextShapelyBox.bounds[2] > maxX):
            maxX = legendTextShapelyBox.bounds[2]
        if (legendTextShapelyBox.bounds[3] > maxY):
            maxY = legendTextShapelyBox.bounds[3]

    unionBbox = box(minX, minY,maxX, maxY)
    return unionBbox

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def isValueText(inputString):
    if not hasNumbers(inputString):
        return False
    inputString = inputString.replace(" ", "") # remove space in the string
    countNumber = sum(c.isdigit() for c in inputString) # count number of digit in string
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

def enlargeShapelyBoxDown(unionLegendShapelyBox,medNumTextBboxHeight,height):
    # output: changed legend shapely box, isSuccessful 
    if unionLegendShapelyBox.bounds[3] + medNumTextBboxHeight > height:
        return unionLegendShapelyBox,False
    enlargedLegendShapelyBox = box(unionLegendShapelyBox.bounds[0],unionLegendShapelyBox.bounds[1],
                            unionLegendShapelyBox.bounds[2],unionLegendShapelyBox.bounds[3] + medNumTextBboxHeight)
    return enlargedLegendShapelyBox,True

def enlargeShapelyBoxUp(unionLegendShapelyBox,medNumTextBboxHeight,height):
    if unionLegendShapelyBox.bounds[1] - medNumTextBboxHeight < 0:
        return unionLegendShapelyBox,False
    enlargedLegendShapelyBox = box(unionLegendShapelyBox.bounds[0],unionLegendShapelyBox.bounds[1] - medNumTextBboxHeight,
                            unionLegendShapelyBox.bounds[2],unionLegendShapelyBox.bounds[3] )
    return enlargedLegendShapelyBox, True

def rectListToShapeBoxList(rectList):
    ShapeBoxList = []
    for rect in rectList:
        minX = rect[0][0][0]/3
        minY = rect[0][0][1]/3
        maxX = rect[0][0][0]/3
        maxY = rect[0][0][1]/3
        for point in rect:
            if point[0][0]/3 < minX:
                minX = point[0][0] /3
            if point[0][0]/3 > maxX:
                maxX = point[0][0]/3
            if point[0][1]/3 < minY:
                minY = point[0][1]/3
            if point[0][1]/3 > maxY:
                maxY = point[0][1]/3
        ShapeBox = box(minX, minY, maxX, maxY)
        ShapeBoxList.append(ShapeBox)
    return ShapeBoxList

def intersectText(rectBox,legendTextShapeBoxList):
    isInterText = False
    for legendTextBox in legendTextShapeBoxList:
        if (rectBox.intersects(legendTextBox)):
            isInterText = True
            break
    return isInterText

def most_common(lst):
    return max(set(lst), key=lst.count)

def reject_outliers(data, mostCommon):
    data = np.array(data)
    return data[abs(data - mostCommon) < 5]

def getRectShapeBox(img, legendShapeBox, legendTextShapeBoxList):
    font = cv2.FONT_HERSHEY_COMPLEX
    # legendPath = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Legend Analysis\\enhanced legend images'
    height = img.shape[0]
    width = img.shape[1]
    enlargeRatio = 3
    dim = (width*3, height*3)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(3,3),0)
    edge = cv2.Canny(gray, 50,200)
    # laplacian = cv2.Laplacian(gray,cv2.CV_8UC1)
    # Taking a matrix of size 5 as the kernel 
    kernel = np.ones((3,3), np.uint8) 

    n= 2
    for i in range(n):
        edge = cv2.dilate(edge, kernel, iterations=1) 
        edge = cv2.erode(edge, kernel, iterations=1) 

    contours, _ = cv2.findContours(
        edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rectList = []  # used to save the rectangles
    rectIndList = []  # save the min max XY value for extraction

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True), True)
        cv2.drawContours(img, [approx], 0, (120, 120, 120), 1)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if len(approx) >=3 and len(approx) <=10:

            test1 = approx[0][0][1]
            test2 = approx[2][0][1]
            if abs(test1 - test2) > 10:
                cv2.putText(img, "Rectangle", (x, y), font, 0.5, (0))
                if x >263 *3 and x < 270 * 3:
                    print(len(rectList))
                rectList.append(approx)
    rectShapeBoxList = rectListToShapeBoxList(rectList)


    # find out all rects intersecting with legendBbox and not intersecting with texts
    # area of legend symbol should be smaller than 1/100 of image area
    legendRectShapeBoxList = []
    for rectBox in rectShapeBoxList:
        isInterText = intersectText(rectBox,legendTextShapeBoxList)
        isInterLegend = legendShapeBox.intersects(rectBox)
        areaRestrict = rectBox.area < (0.01 * (img.shape[0] * img.shape[1])/9)

        if isInterLegend and not isInterText and areaRestrict:
            legendRectShapeBoxList.append(rectBox)

    legendRectShapeBoxList = removeOverlappedBox(legendRectShapeBoxList) # postprocess to remove overlapped rect boxes

    if len(legendRectShapeBoxList) == 0:
        return legendRectShapeBoxList

    # identify whether there is a stardard rect box
    # and standard xMin and xMax
    hasStandRectBox = 0
    XmidList = []
    widthList = []
    XminList = []
    legendRectShapeBoxListUpdated = []

    # get text information
    textHeightList = []
    if len(legendTextShapeBoxList) == 0:
        for rectBox in legendRectShapeBoxList:
            rectHeight = rectBox.bounds[3] - rectBox.bounds[1]
            Xmid = int((rectBox.bounds[0] + rectBox.bounds[2])/2)
            XmidList.append(Xmid)

        mostCommonXmid = most_common(XmidList) 
        XmidList = reject_outliers(XmidList, mostCommonXmid)
        for rectBox in legendRectShapeBoxList:
            Xmid = int((rectBox.bounds[0] + rectBox.bounds[2])/2)
            if Xmid in XmidList:
                legendRectShapeBoxListUpdated.append(rectBox)
        return legendRectShapeBoxListUpdated
        

    for textShapeBox in legendTextShapeBoxList:
        height = abs(textShapeBox.bounds[3] - textShapeBox.bounds[1])
        textHeightList.append(height)
    averageTextHeight = sum(textHeightList)/len(textHeightList)
    modeTextHeight = most_common(textHeightList)

    for rectBox in legendRectShapeBoxList:
        rectHeight = rectBox.bounds[3] - rectBox.bounds[1]
        Xmid = int((rectBox.bounds[0] + rectBox.bounds[2])/2)
        Xmin = int(rectBox.bounds[0])
        width = int(rectBox.bounds[2] + rectBox.bounds[0])
        XmidList.append(Xmid)
        XminList.append(Xmin)
        widthList.append(width)
        # if rect height is larger than text height, the rect include at leat two legend symbols
        # it should be regarded as a standard position
        if rectHeight > 4 * modeTextHeight:
            xMinStand = rectBox.bounds[0]
            xMaxStand = rectBox.bounds[2]
            hasStandRectBox = 1
            break
    if hasStandRectBox ==1:
        verticalAlign = False
        for rectBox in legendRectShapeBoxList:
            xMinRB = rectBox.bounds[0]
            xMaxRB = rectBox.bounds[2]
            verticalAlign = xMinRB < xMaxStand and xMaxRB > xMinStand 
            if verticalAlign:
                legendRectShapeBoxListUpdated.append(rectBox)
    else:
        mostCommonXmid = most_common(XmidList) 
        XmidList = reject_outliers(XmidList, mostCommonXmid)
        for rectBox in legendRectShapeBoxList:
            Xmid = int((rectBox.bounds[0] + rectBox.bounds[2])/2)
            if Xmid in XmidList:
                legendRectShapeBoxListUpdated.append(rectBox)
        
    # cv2.imshow("shapes", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return legendRectShapeBoxListUpdated
    
def removeOverlappedBox(legendRectShapeBoxList):
    # has some problems
    nonOverlapRectBoxList = []
    for rectBox in legendRectShapeBoxList:
        isOverlap = False
        if len(nonOverlapRectBoxList) == 0:
            nonOverlapRectBoxList.append(rectBox)
            continue
        isOverlap = False
        for i in range(len(nonOverlapRectBoxList)):
            if rectBox.intersects(nonOverlapRectBoxList[i]):
                isOverlap = True
                break

        if not isOverlap:
            nonOverlapRectBoxList.append(rectBox)
        elif rectBox.area > nonOverlapRectBoxList[i].area:
            nonOverlapRectBoxList[i] = rectBox

    nonOverlapRectBoxList.sort(key=lambda variable: variable.bounds[3])
    return nonOverlapRectBoxList

def getTextForRect(rect,numerTextBboxes):
    center = rect.centroid
    text = ''
    indexText = -1
    for i in range(len(numerTextBboxes)):
        minY = numerTextBboxes[i][0][0][1]
        maxY = numerTextBboxes[i][0][3][1]
        if center.y <maxY and center.y > minY:
            text = text + numerTextBboxes[i][1]
            indexText = i
            return numerTextBboxes[i], indexText
    if text == '':
        print('No coresponding value for the rect')
    return text, indexText


def main():
    # read detection results from pickle file
    detectResultName = r'C:\Users\jiali\Desktop\MapElementDetection\code\postProcessingDetection\detectResultsFinalBad.pickle'
    with open(detectResultName, 'rb') as fDetectResults:
        detectResults = pickle.load(fDetectResults)

    # read ocr results from pickle file
    ocrResultName = r'C:\Users\jiali\Desktop\MapElementDetection\code\postProcessingDetection\easyOCRFinalBad.pickle'
    with open(ocrResultName, 'rb') as fOCRResults:
        ocrResults = pickle.load(fOCRResults)

    # read image data
    # testImagePath = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\cocoFormatLabeledImages\val'
    testImagePath = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\USStateChoro\finalTestBad'
    testImageDir = os.listdir(testImagePath)
    testImageDir.sort()
    savePath = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\USStateChoro\legendPostProcessFinalBad'
        

    legendResults = []
    # testImageDir = ['map_wm_persons.jpg']
    for imgName in testImageDir:
        postFix = imgName[-4:]
        if postFix == 'json':
            continue
        print(imgName)

    for imgName in testImageDir:
        # imgName = '122.US-map-7custom-pink-red-bigtitle.jpg'

        print('image name: ', imgName)
        postFix = imgName[-4:]
        if postFix == 'json':
            continue
        
        img = cv2.imread(testImagePath + '\\'+imgName)
        height = img.shape[0]
        width = img.shape[1]
        
        # get legend bboxes, text bboxes, and rectangles
        legendBbox = getLegendBboxImage(imgName,detectResults)
        if legendBbox == None:
            # print('no legend detected!')
            # cv2.imshow(imgName, img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cv2.imwrite(savePath + '\\' + imgName, img) 
            continue
        x1Legend = legendBbox[3]
        y1Legend = legendBbox[4]
        x2Legend = legendBbox[5]
        y2Legend = legendBbox[6]
        # box is a type in Shapely for intersection later
        legendShapelyBox = box(x1Legend, y1Legend, x2Legend, y2Legend) 
        # get all text bboxes in the image
        textShapelyBoxList,textBBoxes = getTextBboxes(imgName, ocrResults) 
        if len(textBBoxes) == 0:
            print('No text in the map image!')
            finalLegendBox = legendShapelyBox
                # visualize results
            legendResults.append((imgName,finalLegendBox,[],[],[]))
            startPoint = (int(finalLegendBox.bounds[0]), int(finalLegendBox.bounds[1]))
            endPoint = (int(finalLegendBox.bounds[2]), int(finalLegendBox.bounds[3]))
            cv2.rectangle(img,startPoint,endPoint,(255, 0, 0),2)
            cv2.imwrite(savePath + '\\' + imgName, img) 
            # cv2.imshow(imgName, img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            continue

        # get TextShapelyBoxList and text bboxes in legend
        legendTextShapelyBoxList = getLegendTextShapelyBoxList(legendShapelyBox, textShapelyBoxList)
        legendTextBboxes = getLegendTextBboxes(legendShapelyBox,textBBoxes)
        # if len(legendTextBboxes) == 0:
        #     print('No text in legend of the map image!')
            # cv2.imwrite(savePath + '\\' + imgName, img) 
            # finalLegendBox = legendShapelyBox
            # cv2.imshow(imgName, img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # continue
        numLegendTextShapelyBox = len(legendTextShapelyBoxList)
        # conduct bbox union of legend box and text boxes
        unionLegendShapelyBox = getUnionBbox(legendShapelyBox,legendTextShapelyBoxList) # union of legend text bbox

        # identify the alignment of numeric text bboxes
        align, numerTextShapeBoxList,numerTextBboxes, numTextBboxHeightList = alignmentValueText(legendTextBboxes)
        if align == -1:
            # print('No numerical text in legend of the map image!')
            finalLegendBox = unionLegendShapelyBox
            # cv2.imwrite(savePath + '\\' + imgName, img) 
            # cv2.imshow(imgName, img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # continue
        # print('test')

        #### if alighnment is vertical
        # calculate median heigh of numTextPolygon
        # numTextBboxHeightList.sort()
        # medNumTextBboxHeight = numTextBboxHeightList[int(len(numTextBboxHeightList)/2)]

        # rect detection
        legendRectShapeBoxList = getRectShapeBox(img, unionLegendShapelyBox, legendTextShapelyBoxList)
        numLegendRectShapelyBox = len(legendRectShapeBoxList)
        # print('legend rect numbers: ' + str(numLegendRectShapelyBox))
        unionLegendShapelyBox = getUnionBbox(unionLegendShapelyBox,legendRectShapeBoxList) # union of legend text bbox

        # downward enlarge legend bbox
        # enlargedLegendShapelyBox, isSuccessful = enlargeShapelyBoxDown(unionLegendShapelyBox,medNumTextBboxHeight * 2,height)
        
        # while isSuccessful:
        #     newLegendTextShapelyBoxList = getLegendTextShapelyBoxList(enlargedLegendShapelyBox,textShapelyBoxList)
        #     newLegendRectShapelyBoxList = getRectShapeBox(img,enlargedLegendShapelyBox,textShapelyBoxList)
        #     # if len(newLegendTextShapelyBoxList) == numLegendTextShapelyBox or len(newLegendRectShapelyBoxList) == numLegendRectShapelyBox:
        #     #     break
        #     if len(newLegendTextShapelyBoxList)- numLegendTextShapelyBox == len(newLegendRectShapelyBoxList) - numLegendRectShapelyBox:
        #         numLegendTextShapelyBox = len(newLegendTextShapelyBoxList)
        #         numLegendRectShapelyBox = len(newLegendRectShapelyBoxList)
        #         unionLegendShapelyBox = getUnionBbox(enlargedLegendShapelyBox,newLegendTextShapelyBoxList)
        #         enlargedLegendShapelyBox, isSuccessful = enlargeShapelyBoxDown(unionLegendShapelyBox,medNumTextBboxHeight,height)
        #     else:
        #         break
        #     break


        # upward enlarge legend bbox
        # enlargedLegendShapelyBox, isSuccessful = enlargeShapelyBoxUp(unionLegendShapelyBox,medNumTextBboxHeight,height)
        
        # while isSuccessful:
        #     newLegendTextShapelyBoxList = getLegendTextShapelyBoxList(enlargedLegendShapelyBox,textShapelyBoxList)
        #     newLegendRectShapelyBoxList = getRectShapeBox(img,enlargedLegendShapelyBox,newLegendTextShapelyBoxList)
        #     # if len(newLegendTextShapelyBoxList) == numLegendTextShapelyBox or len(newLegendRectShapelyBoxList) == numLegendRectShapelyBox:
        #     #     break
        #     if len(newLegendTextShapelyBoxList)- numLegendTextShapelyBox == len(newLegendRectShapelyBoxList) - numLegendRectShapelyBox:
        #         numLegendTextShapelyBox = len(newLegendTextShapelyBoxList)
        #         numLegendRectShapelyBox = len(newLegendRectShapelyBoxList)
        #         unionLegendShapelyBox = getUnionBbox(enlargedLegendShapelyBox,newLegendTextShapelyBoxList)
        #         enlargedLegendShapelyBox, isSuccessful = enlargeShapelyBoxUp(unionLegendShapelyBox,medNumTextBboxHeight,height)
        #     else:
        #         break
        # get rect bbox based on legend bbox and legend text bbox
        finalLegendBox = unionLegendShapelyBox # legend box after processed with texts
        # finalLegendBox = legendShapelyBox

        ###########  get legend box processed with legend symbol rectangles
        

        # downward enlarge legend bbox
        # enlargedLegendShapelyBox, isSuccessful = enlargeShapelyBoxDown(unionLegendShapelyBox,medNumTextBboxHeight,height)
        
        # while isSuccessful:
        #     newLegendTextShapelyBoxList = getLegendTextShapelyBoxList(enlargedLegendShapelyBox,textShapelyBoxList)
        #     newLegendRectShapelyBoxList = getRectShapeBox(img,enlargedLegendShapelyBox,legendTextShapelyBoxList)
        #     if len(newLegendRectShapelyBoxList) == numLegendRectShapelyBox or len(newLegendTextShapelyBoxList) == numLegendTextShapelyBox:
        #         break
        #     else:
        #         numLegendRectShapelyBox = len(newLegendRectShapelyBoxList)
        #         numLegendTextShapelyBox = len(newLegendTextShapelyBoxList)
        #         unionLegendShapelyBox = getUnionBbox(enlargedLegendShapelyBox,newLegendRectShapelyBoxList)
        #         enlargedLegendShapelyBox, isSuccessful = enlargeShapelyBoxDown(unionLegendShapelyBox,medNumTextBboxHeight,height)

        
        # finalLegendBox = unionLegendShapelyBox # legend box after processed with texts

        # based on numeric text boxes to complete rectangle detection
        # numerTextShapeBoxList,numerTextBboxes
        # compLegendRectList = []
        # if len(legendRectShapeBoxList) < len(numerTextShapeBoxList):
        #     indexTextRectList = []
        #     indexTextNoRectList = []
        #     # get the rect and corresponding list, store texts with a rect in index list
        #     for rect in legendRectShapeBoxList:
        #         textBBox, indexText = getTextForRect(rect,numerTextBboxes)
        #         compLegendRectList.append([rect, textBBox])
        #         indexTextRectList.append(indexText)
        #     # find the text index without a rect
        #     for i in range(len(numerTextShapeBoxList)):
        #         if i not in indexTextRectList:
        #             indexTextNoRectList.append(i)
        #     ###
        #     # find the average delta x from leftmost point of text bbox to rect, and average size of rect
        #     widthSum, heightSum, deltaXSum = 0,0,0
        #     for rect,textBBox in compLegendRectList:
        #         width = rect.bounds[2] - rect.bounds[0]
        #         height = rect.bounds[3] - rect.bounds[1]
        #         deltaX = textBBox[0][0][0] - rect.centroid.x
        #         widthSum += width
        #         heightSum += height
        #         deltaXSum += deltaX
        #     avgWidth = widthSum / (len(compLegendRectList)+0.000001)
        #     avgHeight = heightSum / (len(compLegendRectList)+0.000001)
        #     avgDelataX = deltaXSum / len(compLegendRectList)
        #     # for each text without a rect, build a rect
        #     for i in range(len(indexTextNoRectList)):
        #         textBbox = numerTextBboxes[indexTextNoRectList[i]]
        #         centerX = textBbox[0][0][0] - avgDelataX
        #         centerY = (textBbox[0][1][1] + textBbox[0][2][1])/2
        #         minX = int (centerX - avgWidth / 2)
        #         maxX = int (centerX + avgWidth / 2)
        #         minY = int (centerY - avgHeight / 2)
        #         maxY = int (centerY + avgHeight / 2)
        #         compRect = box(minX, minY, maxX, maxY)
        #         # legendRectShapeBoxList.append(compRect)
        #         compLegendRectList.append([compRect,textBbox])
        # legendResults.append([imgName] + [finalLegendBox.bounds])

        # print('test')
        colorCategoryList = []
        # for legendRect in legendRectShapeBoxList:
        #     startPoint = (int(legendRect.bounds[0]), int(legendRect.bounds[1]))
        #     endPoint = (int(legendRect.bounds[2]), int(legendRect.bounds[3]))
        #     crop_img = img[startPoint[1]:endPoint[1], startPoint[0]:endPoint[0]]
            #print('test')


        # visualize results
        startPoint = (int(finalLegendBox.bounds[0]), int(finalLegendBox.bounds[1]))
        endPoint = (int(finalLegendBox.bounds[2]), int(finalLegendBox.bounds[3]))
        cv2.rectangle(img,startPoint,endPoint,(255, 0, 0),2)
        for legendRect in legendRectShapeBoxList:
            startPoint = (int(legendRect.bounds[0]), int(legendRect.bounds[1]))
            endPoint = (int(legendRect.bounds[2]), int(legendRect.bounds[3]))
            cv2.rectangle(img,startPoint,endPoint,(0, 255, 0),2)

        for TextRect in legendTextShapelyBoxList:
            startPoint = (int(TextRect.bounds[0]), int(TextRect.bounds[1]))
            endPoint = (int(TextRect.bounds[2]), int(TextRect.bounds[3]))
            cv2.rectangle(img,startPoint,endPoint,(0, 0, 255),2)

        cv2.rectangle(img,startPoint,endPoint,(255, 0, 0),2)
        cv2.imwrite(savePath + '\\' + imgName, img) 
        # cv2.imshow(imgName, img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # save the position of the legend rects, texts and contents
        legendResults.append((imgName,finalLegendBox,legendRectShapeBoxList,legendTextShapelyBoxList,legendTextBboxes))
    print('test')
    with open(r'C:\Users\jiali\Desktop\MapElementDetection\code\postProcessingDetection\legendFinalBadResults.pickle', 'wb') as f:
	    pickle.dump(legendResults,f)

        #### No need to crop the image to get the US boundary
        # crop_img = img[startPoint[1]:endPoint[1], startPoint[0]:endPoint[0]]
        # unionLegendShapelyBox = unionLegendShapelyBox
        # legendRectShapeBoxList = getRectShapeBox(crop_img, unionLegendShapelyBox, legendTextShapelyBoxList)

        # cv2.imshow("cropped", crop_img)
        # cv2.waitKey(0)  
        # print('test')

if __name__ == "__main__":    main()