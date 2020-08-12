# test new added choropleth map images
from PIL import Image
import random
import os

path = r"C:\Users\jiali\Desktop\PyTorch-YOLOv3\data\custom\images"

imagesDir = os.listdir(path)
num_total = len(imagesDir)

width = 800
height = 600
num_pixels = width*height
input_size = width*height*3
input_shape = (width, height, 3)

# Get the image data and store data into X_batches and y_batches
data_pair = []
# Read map images from other projections
for imgName in imagesDir:
    fullName = path +'//' + imgName
    img = Image.open(fullName)
    pixel_values = list(img.getdata())
    data_pair.append(pixel_values)


# data_pair_temp=[data_pair[i] for i in range(300,400)]
data_pair_3 = []
for i in range(num_total):
    # print("i:",i)
    pixel_value_list = []
    for j in range(num_pixels):
        # print("j:",j)
        pixels = data_pair[i][j]
        pixel_value_list.append(pixels[0])
        pixel_value_list.append(pixels[1])
        pixel_value_list.append(pixels[2])

print('test')

