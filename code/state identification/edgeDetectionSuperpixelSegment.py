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
from matplotlib import pyplot as plt
import pickle

def edgeDetector(img):
    # img = cv2.imread(testImagePath + '\\'+imgName)
    font = cv2.FONT_HERSHEY_COMPLEX
    # legendPath = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Legend Analysis\\enhanced legend images'
    height = img.shape[0]
    width = img.shape[1]
    enlargeRatio = 1
    dim = (width*enlargeRatio, height*enlargeRatio)
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
    
    cv2.imshow("shapes", edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return edge

def removeText(img, ocrResults):
    for ocr in ocrResults:
        bbox = ocr[0]
        xMin = int(bbox[0][0])
        yMin = int(bbox[0][1])
        xMax = int(bbox[2][0])
        yMax = int(bbox[2][1])
        value = img[yMax][xMax]

        for x in range(xMin,xMax):
            for y in range(yMin,yMax):
                img[y][x] = value
    # cv2.imshow("shapes", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img

def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if array_equal(elem, myarr)), False)

def edgeDetector(img):
    # img = cv2.imread(testImagePath + '\\'+imgName)
    font = cv2.FONT_HERSHEY_COMPLEX
    # legendPath = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Legend Analysis\\enhanced legend images'
    height = img.shape[0]
    width = img.shape[1]
    enlargeRatio = 1
    dim = (width*enlargeRatio, height*enlargeRatio)
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

    # contours, _ = cv2.findContours(
    #     edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # rectList = []  # used to save the rectangles
    # rectIndList = []  # save the min max XY value for extraction

    # for cnt in contours:
    #     approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True), True)
    #     cv2.drawContours(img, [approx], 0, (120, 120, 120), 1)
    #     x = approx.ravel()[0]
    #     y = approx.ravel()[1]

    #     if len(approx) >=3:

    #         test1 = approx[0][0][1]
    #         test2 = approx[2][0][1]
    #         if abs(test1 - test2) > 10:
    #             cv2.putText(img, "Rectangle", (x, y), font, 0.5, (0))
    #             if x >263 *3 and x < 270 * 3:
    #                 print(len(rectList))
    #             rectList.append(approx)
    # rectShapeBoxList = rectListToShapeBoxList(rectList)

    # # find out all rects intersecting with legendBbox and not intersecting with texts
    # legendRectShapeBoxList = []
    # for rectBox in rectShapeBoxList:
    #     isInterText = intersectText(rectBox,legendTextShapeBoxList)
    #     isInterLegend = legendShapeBox.intersects(rectBox)
    #     if isInterLegend and not isInterText:
    #         legendRectShapeBoxList.append(rectBox)

    # legendRectShapeBoxList = removeOverlappedBox(legendRectShapeBoxList) # postprocess to remove overlapped rect boxes
    
    # cv2.imshow("shapes", edge)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return edge

def main():
    # read detection results from pickle file
    detectResultName = r'C:\Users\jiali\Desktop\MapElementDetection\code\postProcessingDetection\detectResultsOrigin.pickle'
    with open(detectResultName, 'rb') as fDetectResults:
        detectResults = pickle.load(fDetectResults)

    # read ocr results from pickle file
    ocrResultName = r'C:\Users\jiali\Desktop\MapElementDetection\code\postProcessingDetection\ocrBoundsListOrigin.pickle'
    with open(ocrResultName, 'rb') as fOCRResults:
        ocrResults = pickle.load(fOCRResults)

    path = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\USStateChoro\originalSize'
    img1Name = 'ch-07-firstmap-06-1.png'

    imgDetectResult = []
    for dr in detectResults:
        if dr[0] == img1Name:
            imgDetectResult = dr

    # read images and remove texts on the images
    img1 = cv2.imread(path + '\\' + img1Name) # Image1 to be matched
    imgGrey = cv2.imread(path + '\\' + img1Name, 0) 
    ocrImg1 = [ocr[1:] for ocr in ocrResults if ocr[0]==img1Name][0]
    img1Proc = removeText(img1,ocrImg1)
    # get edge detection image
    edge1 = edgeDetector(img1Proc)

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
    plt.axis("off")
    # show the plots
    # plt.show()

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
    indexColorMost = colorCounts.index(max(colorCounts))
    bgColorValue = colorValues[indexColorMost]

    # get the corner superpixel from segmentation results
    # maxSegmentationIDs = []
    # minSegmentationIDs = []
    # for segment in segments:
    #     maxSegmentationIDs.append(segment.max)
    #     minSegmentationIDs.append(segment.min)
    maxSegmentationID = np.amax(segments)
    minSegmentationID = np.amin(segments)

    indexSegmentList = []
    for id in range(minSegmentationID,maxSegmentationID + 1):
        results = np.where(segments == id)
        indexSegmentList.append(results)
    

    print('test')



if __name__ == "__main__":    main()