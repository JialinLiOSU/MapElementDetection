# preprocess the images
from PIL import Image
import glob
import os
from shutil import copyfile

pathOrigin=r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\USStateChoro\originalSize'
pathFinalBad=r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\USStateChoro\finalTestBad'
pathFinalGood = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\USStateChoro\finalTest'
originMapImages = os.listdir(pathOrigin)
finalGoodImages = os.listdir(pathFinalGood)

finalBadImages = []

for om in originMapImages:
    if om not in finalGoodImages:
        finalBadImages.append(om)


for fb in finalBadImages:
    copyfile(pathOrigin + '\\' + fb, pathFinalBad + '\\' + fb)
