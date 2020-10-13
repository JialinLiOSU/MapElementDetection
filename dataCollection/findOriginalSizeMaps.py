# find original size map images from resized images

from PIL import Image
import glob
import os
from PIL import ImageChops
import numpy
import pickle

resizePath = r'C:\Users\jiali\Desktop\PyTorch-YOLOv3\data\custom\images'
originalPath = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\original size choro maps'

# Get the image data and store data into X_batches and y_batches
resizedImages = os.listdir(resizePath)
originalImages = os.listdir(originalPath)

width = 800
height = 600

resizedNames = []
resizedArraies = []

for resImg in resizedImages:
    img = Image.open(resizePath + '\\'+resImg).convert('RGB')
    imgArray = numpy.asarray(img)
    resizedNames.append(resImg)
    resizedArraies.append(imgArray)

oriResNamePairs = []
for oriImg in originalImages: 
    img=Image.open(originalPath + '\\' + oriImg).convert('RGB')
    imgResize = img.resize((width, height), Image.ANTIALIAS)
    oriImgArray = numpy.asarray(imgResize)
    for i in range(len(resizedArraies)):
        if (oriImgArray==resizedArraies[i]).all():
        # isDiff = ImageChops.difference(imgResize, resImg).getbbox()
        # if not isDiff:
            resName = resizedNames[i]
            oriResNamePairs.append((oriImg,resName))

with open(r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\oriResNamePairs.pickle', 'wb') as f:
	    pickle.dump(oriResNamePairs,f)
print(len(oriResNamePairs))

    



