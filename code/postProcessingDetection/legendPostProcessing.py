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
    for result in ocrResults:
        if result[0] == imgName:
            textBboxes = result[1:]
            break
    textShapelyBoxList = []
    textBboxesNew = []
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
        return align,numTextShapelyBoxList,numTextBboxHeightList
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
        return align,numTextShapelyBoxList,numTextBboxHeightList
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
    return align,numTextShapelyBoxList,numTextBboxHeightList

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

        if len(approx) >=3:

            test1 = approx[0][0][1]
            test2 = approx[2][0][1]
            if abs(test1 - test2) > 10:
                cv2.putText(img, "Rectangle", (x, y), font, 0.5, (0))
                if x >263 *3 and x < 270 * 3:
                    print(len(rectList))
                rectList.append(approx)
    rectShapeBoxList = rectListToShapeBoxList(rectList)

    # find out all rects intersecting with legendBbox and not intersecting with texts
    legendRectShapeBoxList = []
    for rectBox in rectShapeBoxList:
        isInterText = intersectText(rectBox,legendTextShapeBoxList)
        isInterLegend = legendShapeBox.intersects(rectBox)
        if isInterLegend and not isInterText and rectBox.area < width * height /5:
            legendRectShapeBoxList.append(rectBox)

    # legendRectShapeBoxList = removeOverlappedBox(legendRectShapeBoxList) # postprocess to remove overlapped rect boxes
   
        
    # cv2.imshow("shapes", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return legendRectShapeBoxList
    
def removeOverlappedBox(legendRectShapeBoxList):
    # has some problems
    newLegendRectBoxList = []
    for rectBox1 in legendRectShapeBoxList:
        for rectBox2 in legendRectShapeBoxList:
            if rectBox1.bounds == rectBox2.bounds:
                continue
            if rectBox1.intersects(rectBox2):
                minX = min(rectBox1.bounds[0],rectBox2.bounds[0])
                minY = min(rectBox1.bounds[1],rectBox2.bounds[1])
                maxX = max(rectBox1.bounds[2],rectBox2.bounds[2])
                maxY = max(rectBox1.bounds[3],rectBox2.bounds[3])
                newBox = box(minX, minY, maxX, maxY)
                newLegendRectBoxList.append(newBox)
                continue
        newLegendRectBoxList.append(rectBox1)
    return newLegendRectBoxList


def main():
    # read detection results from pickle file
    detectResultName = r'C:\Users\jiali\Desktop\MapElementDetection\code\postProcessingDetection\detectResults.pickle'
    with open(detectResultName, 'rb') as fDetectResults:
        detectResults = pickle.load(fDetectResults)

    # read ocr results from pickle file
    ocrResultName = r'C:\Users\jiali\Desktop\MapElementDetection\code\postProcessingDetection\ocrBoundsList.pickle'
    with open(ocrResultName, 'rb') as fOCRResults:
        ocrResults = pickle.load(fOCRResults)

    # read image data
    testImagePath = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\cocoFormatLabeledImages\val'
    testImageDir = os.listdir(testImagePath)

    titleResults = []

    for imgName in testImageDir:
        imgName = 'ChoImg125.jpg'
        

        print('image name: ', imgName)
        postFix = imgName[-4:-1]
        if postFix == 'jso':
            continue
        
        img = cv2.imread(testImagePath + '\\'+imgName)
        height = img.shape[0]
        width = img.shape[1]
        
        # get legend bboxes, text bboxes, and rectangles
        legendBbox = getLegendBboxImage(imgName,detectResults)
        if legendBbox == None:
            print('no legend detected!')
            cv2.imshow(imgName, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
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
            cv2.imshow(imgName, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            continue

        # get TextShapelyBoxList and text bboxes in legend
        legendTextShapelyBoxList = getLegendTextShapelyBoxList(legendShapelyBox, textShapelyBoxList)
        legendTextBboxes = getLegendTextBboxes(legendShapelyBox,textBBoxes)
        if len(legendTextBboxes) == 0:
            print('No text in legend of the map image!')
            finalLegendBox = legendShapelyBox
            cv2.imshow(imgName, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            continue
        numLegendTextShapelyBox = len(legendTextShapelyBoxList)
        # conduct bbox union of legend box and text boxes
        unionLegendShapelyBox = getUnionBbox(legendShapelyBox,legendTextShapelyBoxList) # union of legend text bbox

        # identify the alignment of numeric text bboxes
        align, numTextPolygonList,numTextBboxHeightList = alignmentValueText(legendTextBboxes)
        if align == -1:
            print('No numerical text in legend of the map image!')
            finalLegendBox = unionLegendShapelyBox
            cv2.imshow(imgName, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            continue
        print('test')

        #### if alighnment is vertical
        # calculate median heigh of numTextPolygon
        numTextBboxHeightList.sort()
        medNumTextBboxHeight = numTextBboxHeightList[int(len(numTextBboxHeightList)/2)]

        ###########  get legend box processed with legend symbol rectangles
        # rect detection
        legendRectShapeBoxList = getRectShapeBox(img, unionLegendShapelyBox, textShapelyBoxList)
        numLegendRectShapelyBox = len(legendRectShapeBoxList)
        unionLegendShapelyBox = getUnionBbox(unionLegendShapelyBox,legendRectShapeBoxList) # union of legend text bbox

        # downward enlarge legend bbox
        # enlargedLegendShapelyBox, isSuccessful = enlargeShapelyBoxDown(unionLegendShapelyBox,medNumTextBboxHeight * 2,height)
        
        # while isSuccessful:
        #     newLegendTextShapelyBoxList = getLegendTextShapelyBoxList(enlargedLegendShapelyBox,textShapelyBoxList)
        #     newLegendRectShapelyBoxList = getRectShapeBox(img,enlargedLegendShapelyBox,legendTextShapelyBoxList)
        #     if len(newLegendTextShapelyBoxList) == numLegendTextShapelyBox or len(newLegendRectShapelyBoxList) == numLegendRectShapelyBox:
        #         break
        #     else:
        #         numLegendTextShapelyBox = len(newLegendTextShapelyBoxList)
        #         numLegendRectShapelyBox = len(newLegendRectShapelyBoxList)
        #         unionLegendShapelyBox = getUnionBbox(enlargedLegendShapelyBox,newLegendTextShapelyBoxList)
        #         enlargedLegendShapelyBox, isSuccessful = enlargeShapelyBoxDown(unionLegendShapelyBox,medNumTextBboxHeight,height)
        # upward enlarge legend bbox
        # enlargedLegendShapelyBox, isSuccessful = enlargeShapelyBoxUp(unionLegendShapelyBox,medNumTextBboxHeight,height)
        # while isSuccessful:
        #     newLegendTextShapelyBoxList = getLegendTextShapelyBoxList(enlargedLegendShapelyBox,textShapelyBoxList)
        #     newLegendRectShapelyBoxList = getRectShapeBox(img,enlargedLegendShapelyBox,legendTextShapelyBoxList)
        #     if len(newLegendTextShapelyBoxList) == numLegendTextShapelyBox or len(newLegendRectShapelyBoxList) == numLegendRectShapelyBox:
        #         break
        #     else:
        #         numLegendTextShapelyBox = len(newLegendTextShapelyBoxList)
        #         numLegendRectShapelyBox = len(newLegendRectShapelyBoxList)
        #         unionLegendShapelyBox = getUnionBbox(enlargedLegendShapelyBox,newLegendTextShapelyBoxList)
        #         enlargedLegendShapelyBox, isSuccessful = enlargeShapelyBoxUp(unionLegendShapelyBox,medNumTextBboxHeight,height)
        # get rect bbox based on legend bbox and legend text bbox
        finalLegendBox = unionLegendShapelyBox # legend box after processed with texts

        

        # downward enlarge legend bbox
        # enlargedLegendShapelyBox, isSuccessful = enlargeShapelyBoxDown(unionLegendShapelyBox,medNumTextBboxHeight,height)
        
        # while isSuccessful:
        #     newLegendRectShapelyBoxList = getRectShapeBox(img,enlargedLegendShapelyBox,legendTextShapelyBoxList)
        #     newLegendTextShapelyBoxList = getLegendTextShapelyBoxList(enlargedLegendShapelyBox,textShapelyBoxList)
        #     if len(newLegendRectShapelyBoxList) == numLegendRectShapelyBox or len(newLegendTextShapelyBoxList) == numLegendTextShapelyBox:
        #         break
        #     else:
        #         numLegendRectShapelyBox = len(newLegendRectShapelyBoxList)
        #         numLegendTextShapelyBox = len(newLegendTextShapelyBoxList)
        #         unionLegendShapelyBox = getUnionBbox(enlargedLegendShapelyBox,newLegendRectShapelyBoxList)
        #         enlargedLegendShapelyBox, isSuccessful = enlargeShapelyBoxDown(unionLegendShapelyBox,medNumTextBboxHeight,height)

        
        finalLegendBox = unionLegendShapelyBox # legend box after processed with texts\

        # visualize results
        startPoint = (int(finalLegendBox.bounds[0]), int(finalLegendBox.bounds[1]))
        endPoint = (int(finalLegendBox.bounds[2]), int(finalLegendBox.bounds[3]))
        cv2.rectangle(img,startPoint,endPoint,(255, 0, 0),2)
        cv2.imshow(imgName, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('test')
        

    print('test')
    # with open(r'C:\Users\jiali\Desktop\MapElementDetection\code\postProcessingDetection\titleResults.pickle', 'wb') as f:
	#     pickle.dump(titleResults,f)

if __name__ == "__main__":    main()