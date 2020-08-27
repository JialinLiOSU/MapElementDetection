# generate the label format according to fit the self designed RCNN code
# The input is the labels in format for YOLOv3
# The output is the label annotations for both train and test set

import json
import glob
import os

path = r"C:\Users\jiali\Desktop\PyTorch-YOLOv3\data\custom\labels"
labelSourceDir = os.listdir(path)

pathTraValSource = r"C:\Users\jiali\Desktop\PyTorch-YOLOv3\data\custom"

trainFileName = pathTraValSource + '\\' + 'train.txt'
validFileName = pathTraValSource + '\\' + 'valid.txt'

# read train files to a list for checking
trainFiles = open(trainFileName, 'r')
trainFileList = trainFiles.read().split('\n')
trainFiles.close()
trainNameShortList = []
for trainFileLong in trainFileList:
    if trainFileLong == '' or trainFileLong == ' ':
        continue
    trainNameShort = trainFileLong.split('/')[3]
    trainNameShort = trainNameShort.split('.')[0]
    trainNameShortList.append(trainNameShort)

# read valid files to a list for checking
validFiles = open(validFileName, 'r')
validFileList = validFiles.read().split('\n')
validFiles.close()
validNameShortList = []
for validFileLong in validFileList:
    if validFileLong == '' or validFileLong == ' ':
        continue
    validNameShort = validFileLong.split('/')[3]
    validNameShort = validNameShort.split('.')[0]
    validNameShortList.append(validNameShort)

trainList = []
testList = []
drivePathTrain = '/content/drive/My Drive/Map element detection/cocoFormatDataTrainTest/train'
drivePathTest = '/content/drive/My Drive/Map element detection/cocoFormatDataTrainTest/val'

# Opening JSON file 
for labelfile in labelSourceDir: 
    lfShort = labelfile.split('.')[0]
    if lfShort in trainNameShortList:
        lfOpen = open(path + '\\' + labelfile, "r")
        lfData = lfOpen.read().split('\n')
        lfOpen.close()
        for singleLable in lfData:
            elementList = singleLable.split(' ')
            if elementList[0] != '':
                trainLineStr = drivePathTrain + ',' + elementList[1] + ',' + elementList[2] + ',' + \
                    elementList[3] + ',' + elementList[4] + ',' + elementList[0] + '\n'
                trainList.append(trainLineStr)

    elif lfShort in validNameShortList:
        lfOpen = open(path + '\\' +  labelfile, "r")
        lfData = lfOpen.read().split('\n')
        lfOpen.close()
        for singleLable in lfData:
            elementList = singleLable.split(' ')
            if elementList[0] != '':
                testLineStr = drivePathTrain + ',' + elementList[1] + ',' + elementList[2] + ',' + \
                    elementList[3] + ',' + elementList[4] + ',' + elementList[0] + '\n'
                testList.append(testLineStr)

trainAnnName = pathTraValSource + '\\' + 'trainAnnoSelfDesign.txt'
testAnnName = pathTraValSource + '\\' + 'testAnnoSelfDesign.txt'

file = open(trainAnnName, 'a')
file.writelines(trainList)
file.close()

file = open(testAnnName, 'a')
file.writelines(trainList)
file.close()
    
        


    