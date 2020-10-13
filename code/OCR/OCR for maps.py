# detect texts in images
# show an image
import PIL
from PIL import ImageDraw
import os
import easyocr
import pickle
reader = easyocr.Reader(['en'])

# os.chdir(r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\cocoFormatDataTrainTest\train')
path = "drive/My Drive/Map element detection/dataCollection/cocoFormatDataTrainTest/val"
path = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\cocoFormatDataTrainTest\train'
imageDir = os.listdir(path)

boundsList = []

for imageName in imageDir:
    # im = PIL.Image.open(path + "ChoImg103.jpg")
    postFix = imageName[-4:-1]
    if postFix != 'json':
        bounds = reader.readtext(path + '\\' + imageName)
        boundsList.append([imageName] + bounds)

with open('drive/My Drive/Map element detection/ocrBoundsList.pickle', 'wb') as f:
	  pickle.dump(boundsList,f)

