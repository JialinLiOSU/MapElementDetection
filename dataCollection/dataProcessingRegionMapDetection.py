import numpy as np

import os

from PIL import Image


path = 'C:\\Users\\li.7957\\Desktop\\MapElementDetection\\dataCollection\\new choropleth map\\'

# this size is for transfer learning (VGG16)
width = 800
height = 600
num_pixels = width*height
input_size = width*height*3
input_shape = (width, height, 3)
imageList = []

mapImages = os.listdir(path)
num_images = len(mapImages)

path_target = 'C:\\Users\\li.7957\\Desktop\\MapElementDetection\\dataCollection\\processed choropleth map images\\'
count = 0
for i in range(num_images):
    img = Image.open(path + mapImages[i])
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    if img not in imageList:
        imageList.append(img)
        count = count + 1
        name_target = 'ChoImg'+str(count)+'.jpg'
        img_resized.save(path_target+name_target)

print('number of images', num_images)
print('non repeated image number', len(imageList))
