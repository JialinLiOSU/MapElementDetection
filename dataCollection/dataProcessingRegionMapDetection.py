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

path_target='C:\\Users\\jiali\\Desktop\\MapElementDetection\\dataCollection\\processed new choropleth map\\'

for i in range(num_images):
    img = Image.open(path + newChoMapImgs[i])
    img_resized = img.resize((width, height), Image.ANTIALIAS)
    if img not in imageList:
        imageList.append(img)
        name_target='newChoImg'+str(i+1)+'.jpg'
        img_resized.save(path_target+name_target)

print('number of images',num_images)
print('non repeated image number',len(imageList))



