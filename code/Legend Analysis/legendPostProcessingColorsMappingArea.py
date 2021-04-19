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

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io

# get the most likely legend bbox for imgName from detectResults
def getLegendBboxImage(imgName,legendResults):
    # legendResults.append((imgName,finalLegendBox,legendRectShapeBoxList,legendTextShapelyBoxList,legendTextBboxes))
    legendShapeBbox = None
    legendTextShapelyBoxList = None
    legendTextBboxes = None
    for result in legendResults: # result[0]: image name,
        if result[0] == imgName:
            legendShapeBbox = result[1]
            legendTextShapelyBoxList = result[3]
            legendTextBboxes = result[4]
            break
    
    return legendShapeBbox,legendTextShapelyBoxList,legendTextBboxes

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

def getMostCommonTextHeight(legendTextShapeBoxList):
    textHeightList = []
    for textShapeBox in legendTextShapeBoxList:
        height = abs(textShapeBox.bounds[3] - textShapeBox.bounds[1])
        textHeightList.append(height)
    averageTextHeight = sum(textHeightList)/len(textHeightList)
    modeTextHeight = most_common(textHeightList)
    return modeTextHeight

# find the dominant color value for each legend rectangle
def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]

def getValidXYmidList(legendRectShapeBoxList, imgRGB, colorsMappingArea, greyValuesMappingArea):
    XmidList = []
    YmidList = []
    dominantColorList = []
    dominantColorGreyList = []
    mappingColorIndexList = []
    for rectBox in legendRectShapeBoxList:
        #  should identify whether this rectBox is with mapping area colors
        bounds = rectBox.bounds
        xMin = int(bounds[0])
        yMin = int(bounds[1])
        xMax = int(bounds[2])
        yMax = int(bounds[3])
        croppedImg = imgRGB[yMin:yMax,xMin:xMax]
        dominantColor = unique_count_app(croppedImg)
        rgb_weights = [0.2989, 0.5870, 0.1140]
        dominantColorGrey = int(np.dot(dominantColor, rgb_weights))

        print('test')
        # need to identify whether new value is close to value in list
        diffList = [abs(dominantColorGrey - gv) < 5 for gv in greyValuesMappingArea]

        rectHeight = rectBox.bounds[3] - rectBox.bounds[1]
        Xmid = int((rectBox.bounds[0] + rectBox.bounds[2])/2)
        Ymid = int((rectBox.bounds[1] + rectBox.bounds[3])/2)
        XmidList.append(Xmid)
        YmidList.append(Ymid)
        dominantColorList.append(dominantColor)
        dominantColorGreyList.append(dominantColorGrey)
        if any(diffList):
            mappingColorIndex = diffList.index(True)  
            mappingColorIndexList.append(mappingColorIndex)
        else:
            mappingColorIndexList.append(None)
    return XmidList,YmidList,dominantColorList,dominantColorGreyList,mappingColorIndexList

def getXYmidList(legendRectShapeBoxList):
    XmidList = []
    YmidList = []
    XminList = []
    XmaxList = []
    for rectBox in legendRectShapeBoxList:
        rectHeight = rectBox.bounds[3] - rectBox.bounds[1]
        Xmid = int((rectBox.bounds[0] + rectBox.bounds[2])/2)
        Ymid = int((rectBox.bounds[1] + rectBox.bounds[3])/2)
        Xmin = int(rectBox.bounds[0])
        Xmax = int(rectBox.bounds[2])
        XmidList.append(Xmid)
        YmidList.append(Ymid)
        XminList.append(Xmin)
        XmaxList.append(Xmax)
    return XmidList,YmidList,XminList,XmaxList

def centeroidAddedRect(locations):
    length = locations[0].size
    sum_x = np.sum(locations[1])
    sum_y = np.sum(locations[0])
    return sum_y/length, sum_x/length


def getRectShapeBox(img, legendShapeBox, legendTextShapeBoxList,colorsMappingArea,spLegendResults):
    font = cv2.FONT_HERSHEY_COMPLEX
    # legendPath = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Legend Analysis\\enhanced legend images'
    height = img.shape[0]
    width = img.shape[1]
    enlargeRatio = 3
    dim = (width*3, height*3)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    grayOrigSize = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

    # get corresponding grey values
    greyValuesMappingArea = []
    for color in colorsMappingArea:
        greyValue = rgb2Grey(color)
        greyValuesMappingArea.append(greyValue)

    # find out all rects intersecting with legendBbox and not intersecting with texts
    # area of legend symbol should be smaller than 1/100 of image area
    legendRectShapeBoxListOrigin = []
    for rectBox in rectShapeBoxList:
        if len(legendTextShapeBoxList) == 0:
            isInterText = False
        else:
            isInterText = intersectText(rectBox,legendTextShapeBoxList)
        isInterLegend = legendShapeBox.intersects(rectBox)
        areaRestrict = rectBox.area < (0.01 * (img.shape[0] * img.shape[1])/9)

        if isInterLegend and not isInterText and areaRestrict:
            legendRectShapeBoxListOrigin.append(rectBox)

    legendRectShapeBoxList = removeOverlappedBox(legendRectShapeBoxListOrigin) # postprocess to remove overlapped rect boxes
    # remove rect boxes covering texts
    # legendRectShapeBoxListNonText = []
    # for legendRectShapeBox in legendRectShapeBoxList:
    #     bounds = legendRectShapeBox.bounds
    #     xMinlegendRectShapeBox = int(bounds[0])
    #     yMinlegendRectShapeBox = int(bounds[1])
    #     xMaxlegendRectShapeBox = int(bounds[2])
    #     yMaxlegendRectShapeBox = int(bounds[3])
    #     croppedImg = imgRGB[yMinlegendRectShapeBox:yMaxlegendRectShapeBox, xMinlegendRectShapeBox:xMaxlegendRectShapeBox]
    #     croppedImgGrey = cv2.cvtColor(croppedImg, cv2.COLOR_RGB2GRAY)
    #     numBlack = (croppedImgGrey > 250).sum()
    #     if numBlack < croppedImgGrey.size/10:
    #         legendRectShapeBoxListNonText.append(legendRectShapeBox)
    #     print('test')

    # legendRectShapeBoxList = legendRectShapeBoxListNonText
        
    # results from legend super-pixel analysis
    spLegendShapelyBoxList, dominantLegendGreyColorList,(xMinLeg,yMinLeg) = spLegendResults 
    legendRectShapeBoxListUpdated=[]
    legendRectDomiGreyList = []
    isFromSuperPixelList = []
    if len(legendRectShapeBoxList) == 0:
        
        for i in range(len(dominantLegendGreyColorList)):
            spLegendBox = spLegendShapelyBoxList[i]
            xMin = int(spLegendBox.bounds[0] + xMinLeg)
            yMin = int(spLegendBox.bounds[1] + yMinLeg)
            xMax = int(spLegendBox.bounds[2] + xMinLeg)
            yMax = int(spLegendBox.bounds[3] + yMinLeg)
            spLegendBoxAdjusted = box(xMin,yMin,xMax,yMax)
            legendRectShapeBoxListUpdated.append(spLegendBoxAdjusted)
            legendRectDomiGreyList.append(dominantLegendGreyColorList[i])
            isFromSuperPixelList.append(1)
        
        return legendRectShapeBoxListUpdated,legendRectDomiGreyList,isFromSuperPixelList

    # identify whether there is a stardard rect box
    # and standard xMin and xMax
    hasStandRectBox = 0
    
    widthList = []

    # legendRectShapeBoxListUpdated = []

    # get correct legendRectShapeBoxListUpdated if no text is detected in legend
    XmidList,YmidList, XminList,XmaxList = getXYmidList(legendRectShapeBoxList)
    validXmidList,validYmidList, dominantColorList,dominantColorGreyList,mappingColorIndexList = \
        getValidXYmidList(legendRectShapeBoxList, imgRGB , colorsMappingArea, greyValuesMappingArea)
    
    if len(legendTextShapeBoxList) == 0:
        
        mostCommonXmid = most_common(validXmidList) 
        indexMostCommonXmid = XmidList.index(mostCommonXmid)
        bounds = legendRectShapeBoxList[indexMostCommonXmid].bounds
        widthMostCommon = bounds[2] - bounds[0]
        XmidList = reject_outliers(XmidList, mostCommonXmid)
        mostCommonYmid = most_common(validYmidList) 
        YmidList = reject_outliers(YmidList, mostCommonYmid)
        for rectBox in legendRectShapeBoxList:
            dominantGreyRect,dominantColorRect = getDomiGreyRectBox(rectBox.bounds,imgRGB)
            Xmid = int((rectBox.bounds[0] + rectBox.bounds[2])/2)
            if Xmid in XmidList:
                legendRectShapeBoxListUpdated.append(rectBox)
                legendRectDomiGreyList.append(dominantGreyRect)
                isFromSuperPixelList.append(0)
        # legendRectShapeBoxListUpdated: cleaned legend rectangles, but hasn't deal with not detected rect

        
    else:    
        # get most common text height
        modeTextHeight =  getMostCommonTextHeight(legendTextShapeBoxList)   
        XmidList = []
        for rectBox in legendRectShapeBoxList:
            rectHeight = rectBox.bounds[3] - rectBox.bounds[1]
            Xmid = int((rectBox.bounds[0] + rectBox.bounds[2])/2)
            Xmin = int(rectBox.bounds[0])
            width = int(rectBox.bounds[2] + rectBox.bounds[0])
            XmidList.append(Xmid)
            # if rect height is larger than text height, the rect include at least two legend symbols
            # it should be regarded as a standard position
            if rectHeight > 4 * modeTextHeight:
                xMinStand = rectBox.bounds[0]
                xMaxStand = rectBox.bounds[2]
                
                widthStand = xMaxStand - xMinStand
                hasStandRectBox = 1
                break
        mostCommonXmid = most_common(XmidList) 
        indexMostCommonXmid = XmidList.index(mostCommonXmid)
        bounds = legendRectShapeBoxList[indexMostCommonXmid].bounds
        widthMostCommon = bounds[2] - bounds[0]
        XmidList = reject_outliers(XmidList, mostCommonXmid)
        mostCommonYmid = most_common(validYmidList) 
        YmidList = reject_outliers(YmidList, mostCommonYmid)

        if hasStandRectBox ==1:
            mostCommonXmid = (xMinStand + xMaxStand)/2
            verticalAlign = False
            for rectBox in legendRectShapeBoxList:
                dominantGreyRect,dominantColorRect  = getDomiGreyRectBox(rectBox.bounds,imgRGB)
                xMinRB = rectBox.bounds[0]
                xMaxRB = rectBox.bounds[2]
                widthRB = abs(xMaxRB - xMinRB)
                verticalAlign = xMinRB < xMaxStand and xMaxRB > xMinStand 
                if verticalAlign and widthRB > widthStand/2 and widthRB < widthStand *2:
                    legendRectShapeBoxListUpdated.append(rectBox)
                    legendRectDomiGreyList.append(dominantGreyRect)
                    isFromSuperPixelList.append(0)
        else:
           
            for rectBox in legendRectShapeBoxList:
                dominantGreyRect,dominantColorRect = getDomiGreyRectBox(rectBox.bounds,imgRGB)
                Xmid = int((rectBox.bounds[0] + rectBox.bounds[2])/2)
                if Xmid in XmidList:
                    legendRectShapeBoxListUpdated.append(rectBox)
                    legendRectDomiGreyList.append(dominantGreyRect)
                    isFromSuperPixelList.append(0)


    ##################    adding rects based on mapping area colors    #################
    # from superpixel boxes from legend processing, select valid ones to add to the final rect boxes

    validSpLegendBoxList = []
    # calculate average width of legendRectShapeBoxListUpdated
    sum_width = 0
    # sum_height = 0
    for rectBox in legendRectShapeBoxListUpdated:
        sum_width += abs(rectBox.bounds[2] - rectBox.bounds[0])
    aveWidthLegendRect = sum_width/len(legendRectShapeBoxListUpdated)
        
    # remove None from the list
    mappingColorIndexListClean = [i for i in mappingColorIndexList if i]
    leftColorIndexes = np.setdiff1d(list(range(len(greyValuesMappingArea))),mappingColorIndexListClean)

    leftColorList = []
    leftGreyList = []
    for i in range(leftColorIndexes.size):
        leftColorList.append(colorsMappingArea[leftColorIndexes[i]])
        leftGreyList.append(greyValuesMappingArea[leftColorIndexes[i]])

    for leftGrey in leftGreyList:
        diffList = [abs(leftGrey - dg)<5 for dg in dominantLegendGreyColorList]
        if any(diffList): # there is superpixel with the missing pixel grey
            # add corresponding super pixel bbox into legendRectShapeBoxListUpdated
            indexSPLegShpBoxList = diffList.index(True)
            spLegendBox = spLegendShapelyBoxList[indexSPLegShpBoxList]
            xMin = spLegendBox.bounds[0] + xMinLeg
            yMin = spLegendBox.bounds[1] + yMinLeg
            xMax = spLegendBox.bounds[2] + xMinLeg
            yMax = spLegendBox.bounds[3] + yMinLeg
            xMid = (xMin + xMax)/2
            yMid = (yMin + yMax)/2
            spLegendBoxAdjusted = box(xMin,yMin,xMax,yMax)
            # identify whether most area of the bbox is covered by others
            isValSpBox = isValidSpLegendBox(legendRectShapeBoxListUpdated,spLegendBoxAdjusted)
            # isValSpBox = True
            if (abs(xMid - mostCommonXmid)< widthMostCommon/2 or abs(yMid - mostCommonYmid)<5) and isValSpBox :
                validSpLegendBoxList.append(spLegendBoxAdjusted)
                legendRectDomiGreyList.append(dominantLegendGreyColorList[indexSPLegShpBoxList])
                isFromSuperPixelList.append(1)

    # add legend rects from rect detection and from SP of legend
    legendRectShapeBoxListUpdated += validSpLegendBoxList 

    return legendRectShapeBoxListUpdated,legendRectDomiGreyList,isFromSuperPixelList
        
    # cv2.imshow("shapes", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # return legendRectShapeBoxListUpdated
def isValidSpLegendBox(legendRectShapeBoxListUpdated,spLegendBoxAdjusted):
    # identify whether most area of the bbox is covered by others
    leftArea = spLegendBoxAdjusted.area
    for legendRectShapeBox in legendRectShapeBoxListUpdated:
        if spLegendBoxAdjusted.intersects(legendRectShapeBox) == True:
            intersectArea = spLegendBoxAdjusted.intersection(legendRectShapeBox).area
            leftArea -= intersectArea
        if leftArea < 0 :
            break
    if leftArea > spLegendBoxAdjusted.area*1/2:
        return True
    else:
        return False

    
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

def getColorsMappingArea(imgName,colorsMappingAreaResults):
    for result in colorsMappingAreaResults:
        if result[0] == imgName:
            return result[1]
    return None

def removeText(imgRGBLegend,legendTextShapelyBoxList,dominantColorLegend,xMinLeg,yMinLeg):
    for legTextBox in legendTextShapelyBoxList:
        xMin = int(legTextBox.bounds[0]) - xMinLeg
        yMin = int(legTextBox.bounds[1]) - yMinLeg
        xMax = int(legTextBox.bounds[2]) - xMinLeg
        yMax = int(legTextBox.bounds[3]) - yMinLeg

        for x in range(xMin,xMax):
            for y in range(yMin,yMax):
                imgRGBLegend[y][x] = dominantColorLegend
    # cv2.imshow("shapes", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return imgRGBLegend

def rgb2Grey(dominantColor):
    rgb_weights = [0.2989, 0.5870, 0.1140]
    dominantColorGrey = int(np.dot(dominantColor, rgb_weights))
    return dominantColorGrey

def getDomiGreyRectBox(bounds,imgRGB):
    xMin = int(bounds[0])
    yMin = int(bounds[1])
    xMax = int(bounds[2])
    yMax = int(bounds[3])
    imgRGBRect = imgRGB[yMin:yMax, xMin:xMax]
    dominantColorRect = unique_count_app(imgRGBRect)
    dominantGreyRect = rgb2Grey(dominantColorRect)
    return dominantGreyRect,dominantColorRect

def getSPLegendNonBackground(legendShapelyBbox,imgRGB,legendTextShapelyBoxList):
    # get dominant color in legend area
    xMin = int(legendShapelyBbox.bounds[0])
    yMin = int(legendShapelyBbox.bounds[1])
    xMax = int(legendShapelyBbox.bounds[2])
    yMax = int(legendShapelyBbox.bounds[3])
    imgRGBLegend = imgRGB[yMin:yMax, xMin:xMax]
    dominantColorLegend = unique_count_app(imgRGBLegend)
    dominantGreyLegend = rgb2Grey(dominantColorLegend)
        
    # remove texts in legend   
    if len(legendTextShapelyBoxList) != 0:
        imgRGBLegendClean = removeText(imgRGBLegend,legendTextShapelyBoxList,dominantColorLegend,xMin,yMin)
    else:
        imgRGBLegendClean = imgRGBLegend
    imgGreyLegendClean = cv2.cvtColor(imgRGBLegendClean, cv2.COLOR_RGB2GRAY)

    # cv2.imshow('test', imgGreyLegendClean)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # loop over the number of segments
    # apply SLIC and extract (approximately) the supplied number
    # of segments
    imgRGBLegendCleanFloat = img_as_float(imgRGBLegendClean)
    numSegments = 50
    # get segments from the segmentation results
    segments = slic(imgRGBLegendCleanFloat, n_segments = numSegments, sigma = 5)
    # edgeSegments = edgeDetectorGrey(segments)
    # show the output of SLIC
    fig = plt.figure("Superpixels -- %d segments" % (numSegments))
    ax = fig.add_subplot(1, 1, 1)
    bounds = mark_boundaries(imgRGBLegendCleanFloat, segments)
    ax.imshow(bounds)
    # plt.show()

    # get the list of pairs of coords of pixels with a specific superpixel segmentid
    # index of the list means superpixel segmentid
    maxSegmentationID = np.amax(segments)
    minSegmentationID = np.amin(segments)

    coordPairsList = [] 
    for id in range(minSegmentationID,maxSegmentationID + 1):
        results = np.where(segments == id)
        coordPairs = np.asarray(results).T.tolist()
        coordPairsList.append(coordPairs)

    # identify whether the superpixel is with bg color
    mapRegionSuperPixels = []
    dominantLegendGreyColorList = []
    for coordPairs in coordPairsList:
        colorValueList = []
        for coordPair in coordPairs:
            colorValue = imgGreyLegendClean[coordPair[0],coordPair[1]]
            colorValueList.append(colorValue)
                
        maxOccurValue = max(colorValueList,key=colorValueList.count)
        if abs(maxOccurValue - dominantGreyLegend) > 5:
            mapRegionSuperPixels.append(coordPairs)
            dominantLegendGreyColorList.append(maxOccurValue)

    # generate bboxes of the super pixels
    spLeggendShapelyBoxList = []
    for sp in mapRegionSuperPixels:
        maxCoordSpBbox = np.amax(sp,0)
        minCoordSpBbox = np.amin(sp,0)
        yMaxSpBbox, xMaxSpBbox = maxCoordSpBbox[0],maxCoordSpBbox[1]
        yMinSpBbox, xMinSpBbox = minCoordSpBbox[0], minCoordSpBbox[1]
        spLeggendShapelyBoxList.append(box(xMinSpBbox, yMinSpBbox, xMaxSpBbox, yMaxSpBbox))
    # till now, spLeggendShapelyBoxList and dominantLegendGreyColorList are used 
    return spLeggendShapelyBoxList,dominantLegendGreyColorList,(xMin,yMin)


def main():

    
    # read results from legend post-processing
    # legendResults.append((imgName,finalLegendBox,legendRectShapeBoxList,legendTextShapelyBoxList,legendTextBboxes))
    # with open(r'C:\Users\jiali\Desktop\MapElementDetection\code\postProcessingDetection\legendFinalGoodResultsNew.pickle', 'rb') as f:
    with open(r'C:\Users\jiali\Desktop\MapElementDetection\code\postProcessingDetection\legendFinalShuaichen.pickle', 'rb') as f:
	    legendResults = pickle.load(f)
    legendPostProcessMappingColorResults = []

    # read results from colorsMappingAreaResults to get mapping area colors
    # colorsMappingAreaResults.append((img1Name, superPixelValueList,superPixelValueList))
    # with open(r'C:\Users\jiali\Desktop\MapElementDetection\code\Legend Analysis\colorsMappingAreaResultsFinalGood.pickle', 'rb') as f:
    with open(r'C:\Users\jiali\Desktop\MapElementDetection\code\Legend Analysis\colorsMappingAreaResultsShuaichen.pickle', 'rb') as f:
	    colorsMappingAreaResults = pickle.load(f)
 
    # read image data
    # testImagePath = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\USStateChoro\finalTest'
    testImagePath = r'C:\Users\jiali\Desktop\shuaichen\images'
    testImageDir = os.listdir(testImagePath)
    testImageDir.sort()
    # savePath = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\USStateChoro\legendPostProcessFinalGood'
    savePath = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\USStateChoro\legendPostProcessFinalBad'
        

    for imgName in testImageDir:

        print('image name: ', imgName)
        postFix = imgName[-4:]
        if postFix == 'json':
            continue

        # if imgName != '131.choropleth.jpg':
        #     continue
        # read images and remove texts on the images
        try:
            img = cv2.imread(testImagePath + '\\'+imgName)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            print("not working image: " + imgName + '\n')
            continue

        if img.shape[0] > 1000 or img.shape[1] > 1000:
            print("large image: " + imgName + '\n')
            continue
        
        height = img.shape[0]
        width = img.shape[1]
        
        # get legend bboxes, text bboxes, and rectangles
        legendShapelyBbox,legendTextShapelyBoxList,legendTextBboxes = getLegendBboxImage(imgName,legendResults)
        if legendShapelyBbox == None:
            print('no legend detected!')
            # cv2.imshow(imgName, img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # cv2.imwrite(savePath + '\\' + imgName, img) 
            continue
        
        # if len(legendTextShapelyBoxList) == 0:
        #     print('No text in the legend!')
        #     finalLegendBox = legendShapelyBbox
        #         # visualize results
        #     legendResults.append((imgName,finalLegendBox,[],[],[]))
        #     startPoint = (int(finalLegendBox.bounds[0]), int(finalLegendBox.bounds[1]))
        #     endPoint = (int(finalLegendBox.bounds[2]), int(finalLegendBox.bounds[3]))
        #     cv2.rectangle(img,startPoint,endPoint,(255, 0, 0),2)
        #     # cv2.imwrite(savePath + '\\' + imgName, img) 
        #     # cv2.imshow(imgName, img)
        #     # cv2.waitKey(0)
        #     # cv2.destroyAllWindows()
        #     continue

        ############     get super-pixels of cleaned legend region     ############    
        spLegendShapelyBoxList, dominantLegendGreyColorList,(xMinLeg,yMinLeg) = getSPLegendNonBackground(legendShapelyBbox,imgRGB,legendTextShapelyBoxList)
        spLegendResults = (spLegendShapelyBoxList, dominantLegendGreyColorList,(xMinLeg,yMinLeg))
        # identify the alignment of numeric text bboxes
        # align, numerTextShapeBoxList,numerTextBboxes, numTextBboxHeightList = alignmentValueText(legendTextBboxes)

        # get colors of mapping area from colorsMappingAreaResults
        colorsMappingArea = getColorsMappingArea(imgName,colorsMappingAreaResults)
        
        # rect detection
        legendRectShapeBoxList,legendRectDomiGreyList,isFromSuperPixelList = getRectShapeBox(img, legendShapelyBbox, legendTextShapelyBoxList,colorsMappingArea,spLegendResults)
        numLegendRectShapelyBox = len(legendRectShapeBoxList)

        finalLegendBox = legendShapelyBbox
        
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
        legendPostProcessMappingColorResults.append((imgName,finalLegendBox,legendRectShapeBoxList,legendRectDomiGreyList,isFromSuperPixelList,legendTextShapelyBoxList,legendTextBboxes))
    # print('test')
    with open(r'C:\Users\jiali\Desktop\MapElementDetection\code\Legend Analysis\legendPostProcessMappingColorResultsShuaichen.pickle', 'wb') as f:
	    pickle.dump(legendPostProcessMappingColorResults,f)

        #### No need to crop the image to get the US boundary
        # crop_img = img[startPoint[1]:endPoint[1], startPoint[0]:endPoint[0]]
        # unionLegendShapelyBox = unionLegendShapelyBox
        # legendRectShapeBoxList = getRectShapeBox(crop_img, unionLegendShapelyBox, legendTextShapelyBoxList)

        # cv2.imshow("cropped", crop_img)
        # cv2.waitKey(0)  
        # print('test')

if __name__ == "__main__":    main()