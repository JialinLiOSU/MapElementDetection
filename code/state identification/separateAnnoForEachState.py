# process US state annotation data to extract annotations 
# for template matching testing

import os
rootPath = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\annotatedUSStates'

annotationFileName = 'templateMatchingTestingValidAnnotation.txt'
annotationFileFullName = rootPath + '\\' + annotationFileName

# read image data
testImagePath = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\annotatedUSStates\template matching testing\images'
testImageNameList = os.listdir(testImagePath)
testImageNameList.sort()

# read train files to a list for checking
annotationFile = open(annotationFileFullName, 'r')
annotationList = annotationFile.read().split('\n')
annotationFile.close()

for testImageName in testImageNameList:
    validAnnoList = []
    for annoLine in annotationList:
        annoElements = annoLine.split(',')
        imageName = annoElements[0]
        if (imageName == testImageName) :
            validAnnoList.append(annoLine + '\n')

    filename = testImageName + '.txt'
    file = open(rootPath +'\\'+ filename,'a')
    file.writelines(validAnnoList)
    # file.writelines(incorrectImgNameStrList)
    file.close() 
