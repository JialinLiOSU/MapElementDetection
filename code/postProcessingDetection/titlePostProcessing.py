# The code is for title (legend) detection post processing
# General solution for titles: Do text detection using easyOCR
# if there are some overlap between text object and title object, do union and enlarge bbox
# Dilate in vertical direction by 1 * line height iteratively, until there is no text objects
import pickle
import os
# import easyocr
# reader = easyocr.Reader(['en']) # set OCR for English recognition
from shapely.geometry import Polygon


# # collect the image file names
# imageFileNames = []
# for result in detectResults:
#     imageName = result[0]
#     if imageName not in imageFileNames:
#         imageFileNames.append(imageName)

# get the most likely title bbox for imgName from detectResults
def getTitleBboxImage(imgName,detectResults):
    bboxes = []
    for result in detectResults: # result[0]: image name, result[1]: label
        if result[0] == imgName and result[1] == '0':
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
            return result[1:]

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
        print('image name: ', imgName)
        postFix = imgName[-4:-1]
        if postFix == 'jso':
            continue
        
        # get title bboxes and text bboxes
        titleBbox = getTitleBboxImage(imgName,detectResults)
        if titleBbox == None:
            continue
        x1Title = titleBbox[3]
        y1Title = titleBbox[4]
        x2Title = titleBbox[5]
        y2Title = titleBbox[6]

        textBboxes = getTextBboxes(imgName, ocrResults)

        # create polygon object for title bbox and texts
        titlePolygon = Polygon([(x1Title, y1Title), (x2Title, y1Title), (x2Title, y2Title),(x1Title,y2Title)])
        titleTextBboxes = []
        for bound in textBboxes:
            textPolygon = Polygon([(bound[0][0][0],bound[0][0][1]),(bound[0][1][0],bound[0][1][1]),(bound[0][2][0],bound[0][2][1]),(bound[0][3][0],bound[0][3][1])])
            if titlePolygon.intersects(textPolygon) and bound[2]>pow(10,-10):
                titleTextBboxes.append(bound)
                print(bound[1])
        titleResults.append([imgName] + titleTextBboxes)
        

    print('test')
    # with open(r'C:\Users\jiali\Desktop\MapElementDetection\code\postProcessingDetection\titleResults.pickle', 'wb') as f:
	#     pickle.dump(titleResults,f)

if __name__ == "__main__":    main()