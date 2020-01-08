import numpy as np

import os

from PIL import Image


path = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\dataCollection\\new choropleth map\\'

# this size is for transfer learning (VGG16)
width = 800
height = 600
num_pixels = width*height
input_size = width*height*3
input_shape = (width, height, 3)
imageList = []

newChoMapImgs = os.listdir(path)
num_images = len(newChoMapImgs)

for imgName in newChoMapImgs:
    img = Image.open(path + imgName)
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    pixel_values = list(img_resized.getdata())
    if pixel_values not in imageList:
        imageList.append(pixel_values)
print('number of images',num_images)
print('non repeated image number',len(imageList))



