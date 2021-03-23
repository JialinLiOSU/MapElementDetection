# divide images into train and validation sets
import os
import random

path = r"C:\Users\jiali\Desktop\PyTorch-YOLOv3\data\custom\finalTestBad"

imageNames = os.listdir(path)

# Read map images from other projections
imgNameList = []
for imgName in imageNames:
    if imgName[-4:] != 'json':
        imgNameList.append(imgName)

numImages = len(imgNameList)

validImages = imgNameList

validFilename = r'C:\Users\jiali\Desktop\PyTorch-YOLOv3\data\custom\valid_final_bad.txt'
preFilePath = 'data/custom/images/'

file = open(validFilename, 'a')
for validImageName in validImages:
    file.writelines(preFilePath + validImageName + '\n')
file.close()