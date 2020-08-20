# delete the bad images for R-CNN based models
import json
import glob
import os

pathTarget = r"C:\Users\jiali\Desktop\MapElementDetection\dataCollection\labeledMapsWithCategory\images"
pathRefer = r"C:\Users\jiali\Desktop\PyTorch-YOLOv3\data\custom\images"

# importing the module 
# Get the image data and store data into X_batches and y_batches
targetDir = os.listdir(pathTarget)
referDir = os.listdir(pathRefer)

imgNameList = [] # with no .jpg or .png postfix
# Read map images from refer direction
for imgName in referDir:
    imgNameShort = imgName.split('.')[0]
    imgNameList.append(imgNameShort)

for fileName in targetDir:
    fileNameShort = fileName.split('.')[0]
    if not fileNameShort in imgNameList:
        os.remove(pathTarget + '\\' + fileName)
print("finished")




    
        


    