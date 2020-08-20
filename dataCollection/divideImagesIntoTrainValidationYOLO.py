# divide images into train and validation sets
import os
import random

path = r"C:\Users\jiali\Desktop\PyTorch-YOLOv3\data\custom\images"

imagesToDivide = os.listdir(path)

# Read map images from other projections
imgNameList = []
for imgName in imagesToDivide:
    imgNameList.append(imgName)

numImages = len(imgNameList)

random.shuffle(imgNameList)

numTrain = int(numImages * 4/5)

trainImages = imgNameList[0:numTrain]
validImages = imgNameList[numTrain:numImages]

trainFilename = r'C:\Users\jiali\Desktop\PyTorch-YOLOv3\data\custom\train.txt'
validFilename = r'C:\Users\jiali\Desktop\PyTorch-YOLOv3\data\custom\valid.txt'
preFilePath = 'data/custom/images/'

file = open(trainFilename, 'a')
for trainImageName in trainImages:
    file.writelines(preFilePath + trainImageName + '\n')
file.close()

file = open(validFilename, 'a')
for validImageName in validImages:
    file.writelines(preFilePath + validImageName + '\n')
file.close()