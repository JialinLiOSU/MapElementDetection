# The goal is to identify US states 
# test achieve this goal in two steps
# 1. edge detection for the original images
# 2. feature matching by SIFT descriptor
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io

from numpy import array, array_equal, allclose
import numpy as np
import cv2
from colorthief import ColorThief
from matplotlib import pyplot as plt
import pickle
import numpy as np
import pickle
import os
import sys
sys.path.append(r'C:\Users\jiali\Desktop\Map_Identification_Classification\world map generation\getCartoCoordExtent')
# from shapex import *
# from geom.point import *
from shapely.geometry import Polygon
from shapely.geometry import box
from shapely.geometry import Point
from collections import deque
import random

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
# find the dominant color value for each legend rectangle
def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]
def removeTitleLegend(img1,imgDetectResult):
    imgName = imgDetectResult[0]
    finalLegendBox = imgDetectResult[1]
    xMin = int(finalLegendBox.bounds[0])
    xMax = int(finalLegendBox.bounds[2])
    yMin = int(finalLegendBox.bounds[1])
    yMax = int(finalLegendBox.bounds[3])
    xMin = max(xMin, 0) 
    xMax = min(xMax-1, img1.shape[1])
    yMin = max(yMin, 0)
    yMax = min(yMax-1, img1.shape[0])
    crop_img = img1[yMin:yMax, xMin:xMax]
    dominantColor = unique_count_app(crop_img)
    value = dominantColor
    for x in range(xMin,xMax):
            for y in range(yMin,yMax):
                img1[y][x] = value
    return img1

def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if array_equal(elem, myarr)), False)

def getBackgroundColor(img1,imgGrey):
    # pick up background color
    (height, width, channel) = img1.shape
    heightList = range(10, height - 10, int((height - 20)/10))
    widthList = range(10, width - 10, int((width - 20)/10))
    samplePoints1 = [[10, w] for w in widthList]
    samplePoints2 = [[height - 10, w] for w in widthList]
    samplePoints3 = [[h, 10] for h in heightList]
    samplePoints4 = [[h, width - 10] for h in heightList]
    samplePoints = samplePoints1 + samplePoints2 + samplePoints3 + samplePoints4

    colorValues = []
    colorCounts = []
    for sp in samplePoints:
        colorValue = imgGrey[sp[0],sp[1]]
        if arreq_in_list(colorValue, colorValues) == False:
            colorValues.append(colorValue)
            colorCounts.append(1)
        else:
            index = colorValues.index(colorValue)
            colorCounts[index] += 1
    colorCounts.sort()
    indexColorMost = colorCounts.index(colorCounts[0])
    bgColorValue1 = colorValues[indexColorMost]
    bgColorValue2 = None
    if len(colorCounts)>=2:
        indexColorMostSec = colorCounts.index(colorCounts[1])
        bgColorValue2 = colorValues[indexColorMostSec]
    return bgColorValue1,bgColorValue2

def getStateExtent(shp, country):
    for c in shp:
        x = c['properties']
        if c['properties']['NAME'] == country:
            break
    typeGeom = c['geometry']['type']
    coordGeom = c['geometry']['coordinates']
    minLat,maxLat, minLon, maxLon= 999999999, -999999999, 999999999, -999999999
    # if typeGeom != 'MultiPolygon':
    #     coordGeom = [coordGeom]
    
    for poly in coordGeom:
        if typeGeom != 'MultiPolygon':
            poly = [poly]
        tmpMinLon, tmpMaxLon = min(poly[0])[0], max(poly[0])[0]
        tmpMinLat, tmpMaxLat = min(poly[0], key = lambda t: t[1])[1], max(poly[0],key = lambda t: t[1])[1]
        if tmpMinLon < minLon:
            minLon = tmpMinLon
        if tmpMaxLon > maxLon:
            maxLon = tmpMaxLon
        if tmpMinLat < minLat:
            minLat = tmpMinLat
        if tmpMaxLat > maxLat:
            maxLat = tmpMaxLat

    return (minLon + maxLon)/2, (minLat + maxLat)/2
# get the point list of a state from the shapefile
def getPointList(shp, country):
    for c in shp:
        if c['properties']['NAME'] == country:
            # print('test')
            break
    typeGeom = c['geometry']['type']
    coordGeom = c['geometry']['coordinates']

    if typeGeom != 'MultiPolygon':
#         print(coordGeom[0]) 
        coordList = coordGeom[0]
    else:
        lenList = [len(poly[0]) for poly in coordGeom]
#         print(lenList)
        index = lenList.index(max(lenList))
#         print(index)
#         print(coordGeom[index]) 
        coordList = coordGeom[index][0]
    return [ Point(p[0], p[1]) for p in coordList ]

short_state_names = {
    # 'AK': 'Alaska',
    'AL': 'Alabama',
    'AR': 'Arkansas',
    # 'AS': 'American Samoa',
    'AZ': 'Arizona',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        # 'GU': 'Guam',
        # 'HI': 'Hawaii',
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
        # 'MP': 'Northern Mariana Islands',
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
        # 'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        # 'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

def getXofPointWithMinY(usBound,yMinImg):
    for coordXY in usBound:
        if coordXY[0][1] == yMinImg:
            return coordXY[0][0]
    return None

def findIndexIntersectingBboxes(i,spShapelyBoxList):
    intersectBboxes = []
    for j in range(len(spShapelyBoxList)):
        if i == j:
            continue
        if spShapelyBoxList[i].intersects(spShapelyBoxList[j]):
            intersectBboxes.append(j)
    return intersectBboxes

def convertContour2Polygon(contour):
    contourPointList = list(contour)
    polygonPointList = []
    for point in contourPointList:
        y = point[0][0]
        x = point[0][1]
        polygonPointList.append([x, y])
    polygonPointList.append(polygonPointList[0])
    print(polygonPointList[0])
    print(polygonPointList[-1])
    return Polygon(polygonPointList)

# get the most likely legend bbox for imgName from detectResults
def getLegSymbolBboxes(imgName,detectResults):
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

def main():
    # read detection results from pickle file
    detectResultName = r'C:\Users\jiali\Desktop\MapElementDetection\code\postProcessingDetection\legendFinalShuaichen.pickle'
    with open(detectResultName, 'rb') as fDetectResults:
        # imgName,finalLegendBox,legendRectShapeBoxList,legendTextShapelyBoxList,legendTextBboxes))
        legendResults = pickle.load(fDetectResults)

    # read ocr results from pickle file
    ocrResultName = r'C:\Users\jiali\Desktop\MapElementDetection\code\postProcessingDetection\easyOCRFinalBad.pickle'
    with open(ocrResultName, 'rb') as fOCRResults:
        ocrResults = pickle.load(fOCRResults)

    path = r'C:\Users\jiali\Desktop\shuaichen\images'

    colorsMappingAreaResults = []
    
    for legResult in legendResults:
        
        img1Name = legResult[0]
        print('image name: ' + img1Name)
        # if img1Name == '115936742_3317778874932504_6087693444288850821_o.jpg':
        #     continue
        # img = cv2.imread(path + '\\' + img1Name) # Image1 to be matched
        # img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # bgColor,bgColorSec = getBackgroundColor(img1, imgGrey)

        # img1Area = img1.shape[0] *img1.shape[1]
        finalLegendBox = legResult[1]
        legendRectShapeBoxList = legResult[2]
        legendTextShapelyBoxList =legResult[3]
        legendTextBboxes = legResult[4]
        numRectBoxes = len(legendRectShapeBoxList)

        imgDetectResult = []
        for dr in legendResults:
            if dr[0] == img1Name:
                imgDetectResult = dr

        # read images and remove texts on the images
        try:
            img = cv2.imread(path + '\\' + img1Name) # Image1 to be matched
            img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            print("not working image: " + img1Name + '\n')
            continue

        if img.shape[0] > 1000 or img.shape[1] > 1000:
            print("large image: " + img1Name + '\n')
            continue
        
        # imgGrey = cv2.imread(path + '\\' + img1Name, 0) 
        ocrImg1 = [ocr[1:] for ocr in ocrResults if ocr[0]==img1Name]
            
        if len(imgDetectResult) > 1:
            img1NoTL = removeTitleLegend(img1,imgDetectResult)
        else:
            img1NoTL = img1
            
        if len(ocrImg1) != 0:
            ocrImg1 = ocrImg1[0]
            img1Proc = removeText(img1NoTL,ocrImg1)
        else:
            img1Proc = img1NoTL
        imgGrey = cv2.cvtColor(img1Proc, cv2.COLOR_RGB2GRAY)
        # get edge detection image

        # loop over the number of segments
        # apply SLIC and extract (approximately) the supplied number
        # of segments
        image = img_as_float(img1Proc)
        numSegments = 300
        # get segments from the segmentation results
        segments = slic(image, n_segments = numSegments, sigma = 5)
        # edgeSegments = edgeDetectorGrey(segments)
        # show the output of SLIC
        fig = plt.figure("Superpixels -- %d segments" % (numSegments))
        ax = fig.add_subplot(1, 1, 1)
        bounds = mark_boundaries(image, segments)
        ax.imshow(bounds)

        bgColor,bgColorSec = getBackgroundColor(img1Proc, imgGrey)

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
        for coordPairs in coordPairsList:
            colorValueList = []
            for coordPair in coordPairs:
                colorValue = imgGrey[coordPair[0],coordPair[1]]
                colorValueList.append(colorValue)
                
            maxOccurValue = max(colorValueList,key=colorValueList.count)
            if bgColorSec == None:
                if abs(maxOccurValue - bgColor) > 10:
                    mapRegionSuperPixels.append(coordPairs)
            else:
                if abs(maxOccurValue - bgColor) > 10 and abs(maxOccurValue - bgColorSec) > 10:
                    mapRegionSuperPixels.append(coordPairs)

        # generate bboxes of the super pixels
        spShapelyBoxList = []
        for sp in mapRegionSuperPixels:
            maxCoordSpBbox = np.amax(sp,0)
            minCoordSpBbox = np.amin(sp,0)
            xMaxSpBbox, yMaxSpBbox = maxCoordSpBbox[0],maxCoordSpBbox[1]
            xMinSpBbox, yMinSpBbox = minCoordSpBbox[0], minCoordSpBbox[1]
            spShapelyBoxList.append(box(xMinSpBbox, yMinSpBbox,xMaxSpBbox, yMaxSpBbox))

        # from the bbox of each super-pixel, get dominant pixel values
        superPixelGreyValueList = []
        superPixelValueList = []
        for shapelybox in spShapelyBoxList:
            bounds = shapelybox.bounds
            xMin = int(bounds[0])
            yMin = int(bounds[1])
            xMax = int(bounds[2])
            yMax = int(bounds[3])
            croppedImg = img1Proc[xMin:xMax,yMin:yMax]
            dominantColor = unique_count_app(croppedImg)
            rgb_weights = [0.2989, 0.5870, 0.1140]
            dominantColorGrey = int(np.dot(dominantColor, rgb_weights))

            # need to identify whether new value is close to value in list
            diffList = [abs(dominantColorGrey - pv)>10 for pv in superPixelGreyValueList]
            if  all(diffList) and abs(dominantColorGrey - bgColor) > 10:
                if bgColorSec != None and abs(dominantColorGrey - bgColorSec) <= 10:
                    continue
                superPixelGreyValueList.append(dominantColorGrey)
                superPixelValueList.append(dominantColor)
        
        colorsMappingAreaResults.append((img1Name, superPixelValueList,superPixelValueList))

        # get symbols of legend from post processing results of legend
        # legendRectDomPixelValueList = []
        # for legendRectShapeBox in legendRectShapeBoxList:
        #     bounds = legendRectShapeBox.bounds
        #     startPoint = (int(bounds[0]), int(bounds[1]))
        #     endPoint = (int(bounds[2]), int(bounds[3]))
        #     # get pixel array for current legend rect
        #     crop_img = img[startPoint[1]:endPoint[1], startPoint[0]:endPoint[0]]
        #     dominantColorLegRect = unique_count_app(crop_img)
        #     legendRectDomPixelValueList.append(dominantColorLegRect)
        #     print('test')
        # compare legendRectDomPixelValueList and superPixelValueList

    with open(r'C:\Users\jiali\Desktop\MapElementDetection\code\Legend Analysis\colorsMappingAreaResultsShuaichen.pickle', 'wb') as f:
	    pickle.dump(colorsMappingAreaResults,f)

    
if __name__ == "__main__":    main()