# detect texts in images
# show an image
import PIL
from PIL import ImageDraw
import os
os.chdir(r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\cocoFormatDataTrainTest\train')
im = PIL.Image.open("ChoImg192.jpg")

import easyocr
reader = easyocr.Reader(['en'])
bounds = reader.readtext('ChoImg192.jpg')
print(bounds)

# def draw_boxes(image, bounds, color='yellow', width=2):
#     draw = ImageDraw.Draw(image)
#     for bound in bounds:
#         p0, p1, p2, p3 = bound[0]
#         draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
#     return image

# draw_boxes(im, bounds)