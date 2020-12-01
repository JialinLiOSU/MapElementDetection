# The goal is to identify US states 
# test achieve this goal in two steps
# 1. edge detection for the original images
# 2. feature matching by SIFT descriptor

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
    enlargeRatio = 2
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
    cv2.imshow("shapes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img


def main():
    # read detection results from pickle file
    detectResultName = r'C:\Users\jiali\Desktop\MapElementDetection\code\postProcessingDetection\detectResultsOrigin.pickle'
    with open(detectResultName, 'rb') as fDetectResults:
        detectResults = pickle.load(fDetectResults)

    # read ocr results from pickle file
    ocrResultName = r'C:\Users\jiali\Desktop\MapElementDetection\code\postProcessingDetection\ocrBoundsListOrigin.pickle'
    with open(ocrResultName, 'rb') as fOCRResults:
        ocrResults = pickle.load(fOCRResults)

    path = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\original size choro maps'
    img1Name = '35.lab-7.jpg'
    img2Name = '7.7886413.jpg'

    # read images and remove texts on the images
    img1 = cv2.imread(path + '\\' + img1Name) # Image1 to be matched
    ocrImg1 = [ocr[1:] for ocr in ocrResults if ocr[0]==img1Name][0]
    img1Proc = removeText(img1,ocrImg1)

    img2 = cv2.imread(path + '\\' + img2Name) # Image2 to be matched
    ocrImg2 = [ocr[1:] for ocr in ocrResults if ocr[0]==img2Name][0]
    img2Proc = removeText(img2,ocrImg2)

    edge1 = edgeDetector(img1Proc)
    edge2 = edgeDetector(img2Proc)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(edge1,None)
    kp2, des2 = sift.detectAndCompute(edge2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]

    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = 0)

    img3 = cv2.drawMatchesKnn(edge1,kp1,edge2,kp2,matches,None,**draw_params)

    plt.imshow(img3,),plt.show()





if __name__ == "__main__":    main()