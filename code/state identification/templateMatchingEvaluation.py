# process US state annotation data to extract annotations 
# for template matching testing

import os
import pickle
import cv2
import numpy as np

# find the dominant color value for each legend rectangle
def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]
def rgb2Grey(dominantColor):
    rgb_weights = [0.2989, 0.5870, 0.1140]
    dominantColorGrey = int(np.dot(dominantColor, rgb_weights))
    return dominantColorGrey

def main():

    # read template matching results
    indexMethod = 5
    print('indexMethod: '+ str(indexMethod))
    with open(r'C:\Users\jiali\Desktop\MapElementDetection\code\state identification\templateResults' + str(indexMethod) + '.pickle', 'rb') as f:
        resultsImagesList = pickle.load(f)

    # read annotation files
    rootPath = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\annotatedUSStates\template matching testing'
    annoPath = rootPath + '\\' + 'annos'
    # read image data
    testImagePath = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\annotatedUSStates\template matching testing\images'
    testImageNameList = os.listdir(testImagePath)

    validStateNames = ['Alabama','Arkansas','Arizona','California',
        'Colorado','Connecticut','Delaware','Florida','Georgia','Iowa','Idaho','Illinois',
        'Indiana','Kansas','Kentucky','Louisiana','Massachusetts','Maryland','Maine','Michigan','Minnesota','Missouri',
        'Mississippi','Montana','North Carolina','North Dakota','Nebraska','New Hampshire','New Jersey',
        'New Mexico','Nevada','New York','Ohio', 'Oklahoma','Oregon','Pennsylvania','Rhode Island','South Carolina',
        'South Dakota','Tennessee','Texas','Utah','Virginia','Vermont','Washington','Wisconsin','West Virginia','Wyoming']

    validAnnoList = []
    accuracyList = []
    for i in range(len(testImageNameList)):
        # read temp matching results for this image
        tempMatchResult = resultsImagesList[i]
        greyStates = tempMatchResult[0]

        # calculate grey value from annotation
        testImageName = testImageNameList[i]
        img = cv2.imread(testImagePath + '\\' + testImageName,0)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        annotationFileName = testImageName + '.txt'
        annotationFileFullName = annoPath + '\\' + annotationFileName
        annotationFile = open(annotationFileFullName, 'r')
        annotationList = annotationFile.read().split('\n')
        annotationFile.close()
        stateList = []
        stateGreyList = [-1 for i in range(len(validStateNames))]
        for annoLine in annotationList:
            if annoLine =='':
                continue
            annoElements = annoLine.split(',')
            imageName = annoElements[0]
            x1 = float(annoElements[1])
            y1 = float(annoElements[2])
            width = float(annoElements[3])
            height = float(annoElements[4])
            centerY = int(y1 + height / 2)
            centerX = int(x1 + width / 2)
            stateName = annoElements[5]
            if stateName not in stateList:
                stateList.append(stateName)
                centerColor = imgRGB[centerY,centerX]
                stateGrey = rgb2Grey(centerColor)
                indexState = validStateNames.index(stateName)
                stateGreyList[indexState] = stateGrey
        
        sameList = [abs(greyStates[i]-stateGreyList[i])<5 for i in range(len(greyStates))]
        accuracy = sameList.count(True)/len(sameList)
        accuracyList.append(accuracy)
    aveAccuracy = sum(accuracyList)/len(accuracyList)
    print('aveAccuracy:' + str(aveAccuracy))


            
if __name__ == "__main__":    main()

        
