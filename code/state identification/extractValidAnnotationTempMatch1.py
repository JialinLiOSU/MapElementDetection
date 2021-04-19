# process US state annotation data to extract annotations 
# for template matching testing

import os
rootPath = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\annotatedUSStates'

annotationFileName = 'templateMatchingTestingAnnotation.txt'
annotationFileFullName = rootPath + '\\' + annotationFileName

# read image data
testImagePath = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\annotatedUSStates\template matching testing'
testImageNameList = os.listdir(testImagePath)
testImageNameList.sort()

# us state name
validStateNames = ['Alabama','Arkansas','Arizona','California',
    'Colorado','Connecticut','Delaware','Florida','Georgia','Iowa','Idaho','Illinois',
    'Indiana','Kansas','Kentucky','Louisiana','Massachusetts','Maryland','Maine','Michigan','Minnesota','Missouri',
    'Mississippi','Montana','North Carolina','North Dakota','Nebraska','New Hampshire','New Jersey',
    'New Mexico','Nevada','New York','Ohio', 'Oklahoma','Oregon','Pennsylvania','Rhode Island','South Carolina',
    'South Dakota','Tennessee','Texas','Utah','Virginia','Vermont','Washington','Wisconsin','West Virginia','Wyoming']

print('number of valid State names: ' + str(len(validStateNames)))
# read train files to a list for checking
annotationFile = open(annotationFileFullName, 'r')
annotationList = annotationFile.read().split('\n')
annotationFile.close()
validAnnoList = []
stateNameList = []
imageNameList = []
for annoLine in annotationList:
    annoElements = annoLine.split(',')
    imageName = annoElements[0]
    stateName = annoElements[5]
    if (imageName in testImageNameList) and (stateName in validStateNames) :
        stateNameList.append(stateName)
        validAnnoList.append(annoLine + '\n')

filename='templateMatchingTestingValidAnnotation' + '.txt'
file = open(rootPath +'\\'+ filename,'a')
file.writelines(validAnnoList)
# file.writelines(incorrectImgNameStrList)
file.close() 
