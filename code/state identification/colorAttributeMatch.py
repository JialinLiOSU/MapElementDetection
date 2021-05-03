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

def most_common(lst):
    return max(set(lst), key=lst.count)

# find the dominant color value for each legend rectangle
def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]

def findCentroids(imgName, centroidImgCoordList):
    centroidCoord = []
    for centroidImg in centroidImgCoordList:
        if centroidImg[0] == imgName:
            centroidCoord = centroidImg[1]
            return centroidCoord
    return centroidCoord

def removeText(img, ocrResults):
    for ocr in ocrResults:
        bbox = ocr[0]
        xMin = int(bbox[0][0])
        yMin = int(bbox[0][1])
        xMax = int(bbox[2][0])
        yMax = int(bbox[2][1])
        crop_img = img[yMin:yMax, xMin:xMax]
        dominantColor = unique_count_app(crop_img)
        value = dominantColor

        for x in range(xMin,xMax):
            for y in range(yMin,yMax):
                img[y][x] = value
    # cv2.imshow("shapes", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img

def removeTitleLegend(img1,imgDetectResult):
    imgName = imgDetectResult[0]
    finalLegendBox = imgDetectResult[1]
    xMin = int(finalLegendBox.bounds[0])
    xMax = int(finalLegendBox.bounds[2])
    yMin = int(finalLegendBox.bounds[1])
    yMax = int(finalLegendBox.bounds[3])
    xMin = max(xMin, 0) 
    xMax = max(xMax-1, 0)
    yMin = max(yMin, 0)
    yMax = max(yMax-1, 0)
    crop_img = img1[yMin:yMax, xMin:xMax]
    dominantColor = unique_count_app(crop_img)
    value = dominantColor
    for x in range(xMin,xMax):
            for y in range(yMin,yMax):
                img1[y][x] = value
    return img1

short_state_names = {
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

def findIndexCloseColor(dominantColor,refColorList):
    index = 0
    minDist = 999999999999999
    for i in range(len(refColorList)):
        refColor = refColorList[i]
        diffX = int(dominantColor[0]) - int(refColor[0])
        diffY = int(dominantColor[1]) - int(refColor[1])
        diffZ = int(dominantColor[2]) - int(refColor[2])
        dist = diffX**2 + diffY**2 + diffZ**2
        if dist < minDist:
            minDist = dist
            index = i
    return index


def main():
    # read detection results from pickle file
    testImagePath = r'C:\Users\jiali\Desktop\shuaichen\images'
    testImageDir = os.listdir(testImagePath)

    with open(r'C:\Users\jiali\Desktop\MapElementDetection\code\Legend Analysis\legendMatchingResultsShuaichen.pickle', 'rb') as f:
        # legendMatchingResults.append((imgName, dominantColorList,textList))
	    legendMatchingResults = pickle.load(f)
    with open(r'C:\Users\jiali\Desktop\MapElementDetection\code\state identification\centroidImgCoordList.pickle', 'rb') as f:
        # centroidImgCoordList.append((img1Name,centroidStateCoordList))
	    centroidImgCoordList = pickle.load(f)
    
    # read detection results from pickle file
    detectResultName = r'C:\Users\jiali\Desktop\MapElementDetection\code\postProcessingDetection\legendFinalBadResults.pickle'
    with open(detectResultName, 'rb') as fDetectResults:
        detectResults = pickle.load(fDetectResults)

    # read ocr results from pickle file
    ocrResultName = r'C:\Users\jiali\Desktop\MapElementDetection\code\postProcessingDetection\ocrFinalBad.pickle'
    with open(ocrResultName, 'rb') as fOCRResults:
        ocrResults = pickle.load(fOCRResults)

    stateNameList = list(short_state_names.values())

    strList = []  # save the strings to be written in files
    
    

    for legMatResult in legendMatchingResults:
        imgName, refColorList,refTextList = legMatResult
        strTemp = imgName + '\n'
        strList.append(strTemp)

        coordCentroids = findCentroids(imgName, centroidImgCoordList)
        if len(coordCentroids) != 0:
            img = cv2.imread(testImagePath + '\\' + imgName) # Image1 to be matched
            img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # remove texts and titles and legends
            imgDetectResult = []
            for dr in detectResults:
                if dr[0] == imgName:
                    imgDetectResult = dr
            # imgGrey = cv2.imread(path + '\\' + img1Name, 0) 
            ocrImg1 = [ocr[1:] for ocr in ocrResults if ocr[0]==imgName]
            if len(imgDetectResult) > 1:
                img1NoTL = removeTitleLegend(img1,imgDetectResult)
            else:
                img1NoTL = img1
            if len(ocrImg1) != 0:
                ocrImg1 = ocrImg1[0]
                img1Proc = removeText(img1NoTL,ocrImg1)
            else:
                img1Proc = img1NoTL
            
            for i in range(len(stateNameList)):
                coordCentroid = coordCentroids[i]
                x, y = int(coordCentroid[0]), int(coordCentroid[1])
                startPoint = (x - 5, y - 5)
                endPoint = (x + 5, y + 5)
                # get pixel array for current legend rect
                crop_img = img[startPoint[1]:endPoint[1], startPoint[0]:endPoint[0]]
                dominantColor = unique_count_app(crop_img)
                indexCloseColor = findIndexCloseColor(dominantColor,refColorList)
                attributeText = refTextList[indexCloseColor]
                strTemp = stateNameList[i] + ': ' + attributeText + '\n'
                strList.append(strTemp)
                
                # print('test')
    
    filename = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\USStateChoro\stateAttributeTable_FinalBad.txt'
    file = open(filename, 'a')
    file.writelines(strList)
    file.close()

if __name__ == "__main__":    main()

