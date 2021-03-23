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
from scipy.stats import linregress


# def hasNumbers(inputString):
#     return any(char.isdigit() for char in inputString)

# find the dominant color value for each legend rectangle
def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]

def non_increasing(L):
    return all(x>=y for x, y in zip(L, L[1:]))

def non_decreasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))

def monotonic(L):
    return non_increasing(L) or non_decreasing(L)

def incdec_test(x):
    i = np.array(x).argmax()
    if i == 0 or i == len(x)-1:
        return False
    isNonDecrease = non_decreasing(x[0:i])
    isNonDecrease = non_increasing(x[i:])

    return isNonDecrease and isNonDecrease

def identColorScheme(colorHSVList):
    Vlist = [hsv[2] for hsv in colorHSVList]
    isMono = monotonic(Vlist)
    incdec = incdec_test(Vlist)
    if isMono == True:
        # print("Monotonical and sequential")
        return 1
    elif incdec == True:
        # print("Increase and then decrease and diverging")
        return 2
    else:
        # print("qualitative scheme")
        return 0

def identColorBlindSafe(colorHSVList, colorCIEList):
    Vlist = [hsv[2] for hsv in colorHSVList]
    isMono = monotonic(Vlist)
    incdec = incdec_test(Vlist)
    if incdec == True:
        Xlist = [XYZ[0] for XYZ in colorCIEList]
        Ylist = [XYZ[1] for XYZ in colorCIEList]
        Zlist = [XYZ[2] for XYZ in colorCIEList]
        xList = []
        yList = []
        for i in range(len(colorCIEList)):
            X = int(Xlist[i])
            Y = int(Ylist[i])
            Z = int(Zlist[i])
            sum = (X + Y + Z)
            if sum == 0:
                continue
            x = X/sum
            y = Y/sum
            xList.append(x)
            yList.append(y)
        regResults = linregress(xList, yList)
        slope = regResults[0]
        # print('slope: '+ str(slope))
        intcpt = regResults[1]
        # print('intercept: '+ str(intcpt))
        if slope > 0.57:
            # print("colorblind safe")
            return True
        elif slope > 0 and slope <=0.57:
            if intcpt > -0.33*slope + 0.4:
                # print("colorblind safe")
                return True
            else:
                # print("colorblind unsafe")
                return False
        else:
            # print("colorblind unsafe")
            return False
    else:
        return True

def main():
    # read legend post processing results from pickle file
    legendResultsFile = r'C:\Users\jiali\Desktop\MapElementDetection\code\postProcessingDetection\legendFinalGoodResults.pickle'
    with open(legendResultsFile, 'rb') as fDetectResults:
        legendResults = pickle.load(fDetectResults)

    testImagePath = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\USStateChoro\finalTest'
    testImageDir = os.listdir(testImagePath)

    for legendResult in legendResults:
        # legendResult = legendResults[44]
        legendRectShapeBoxList = legendResult[2]
        imgName = legendResult[0]

        if len(legendRectShapeBoxList) <= 1:
            print('Number of legend symbols: '+ str(len(legendRectShapeBoxList)))
            print("Not enough legend symbols for analysis")
            continue

        print('image name: '+ imgName)
        img = cv2.imread(testImagePath + '\\'+imgName)
        
        legendImgList = []
        colorCIEList = []
        colorHSVList = []
        for legendRect in legendRectShapeBoxList:
            startPoint = (int(legendRect.bounds[0]), int(legendRect.bounds[1]))
            endPoint = (int(legendRect.bounds[2]), int(legendRect.bounds[3]))
            # get pixel array for current legend rect
            crop_img = img[startPoint[1]:endPoint[1], startPoint[0]:endPoint[0]]
            legendImgList.append(crop_img)
            # convert pixel array from BGR to HSV and XYZ in CIE model
            hsv_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
            cie_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2XYZ)
            dominantColorHSV = unique_count_app(hsv_img)
            dominantColorCIE = unique_count_app(cie_img) # CIE model is XYZ need to be converted to xyY
            # dominantColorHSV = cv2.cvtColor(dominantColorBGR, cv2.COLOR_BGR2HSV)
            # colorBGRList.append(dominantColorBGR)
            colorHSVList.append(dominantColorHSV)
            colorCIEList.append(dominantColorCIE)
        
        # identify diverging or sequential schemes
        colorSchemeType = identColorScheme(colorHSVList)

        if colorSchemeType == 1:
            print("Monotonical and sequential")
        elif colorSchemeType == 2:
            print("Increase and then decrease and diverging")
        else:
            print("qualitative scheme")

        isColorBlindSafe = identColorBlindSafe(colorHSVList, colorCIEList)
        
        if isColorBlindSafe == True:
            print("colorblind safe")
        else:
            print("colorblind unsafe")

    print('test')

if __name__ == "__main__":    main()