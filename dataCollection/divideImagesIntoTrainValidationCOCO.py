# divide images into train and validation sets
import os
import random

path = r"C:\Users\jiali\Desktop\PyTorch-YOLOv3\data\custom"

trainFileName = path + '\\' + 'train.txt'
validFileName = path + '\\' + 'valid.txt'

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

# check each file in train and valid folder, and delete the files not belonging to the folder
pathTrain = r"C:\Users\jiali\Desktop\MapElementDetection\dataCollection\cocoFormatLabeledImages\train"
pathValid = r"C:\Users\jiali\Desktop\MapElementDetection\dataCollection\cocoFormatLabeledImages\valid"

trainDir = os.listdir(pathTrain)
validDir = os.listdir(pathValid)

for fileName in trainDir:
    fileNameShort = fileName.split('.')[0]
    if not fileNameShort in trainNameShortList:
        os.remove(pathTrain + '\\' + fileName)

for fileName in validDir:
    fileNameShort = fileName.split('.')[0]
    if not fileNameShort in validNameShortList:
        os.remove(pathValid + '\\' + fileName)

print("finished")