# test whether the images and labels are matched
from PIL import Image
import random
import os

pathImage = r"C:\Users\jiali\Desktop\PyTorch-YOLOv3\data\custom\images"
pathLabel = r"C:\Users\jiali\Desktop\PyTorch-YOLOv3\data\custom\labels"

imagesDir = os.listdir(pathImage)
labelDir = os.listdir(pathLabel)
imgNames = []
labelNames = []

for imgName in imagesDir:
    shortName = imgName.split('.')[0]
    imgNames.append(shortName)

for labelName in labelDir:
    shortName = labelName.split('.')[0]
    labelNames.append(labelName)

for imgName in imgNames:
    if imgName in imgNames:
        continue
    else:
        print(imgName)