from PIL import Image
import pytesseract
import sys
import numpy as np


path = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\USStateChoro\finalTest'

image_file = path + '\\' + '7.7886413.jpg'
img = Image.open(image_file).convert("RGB")

print(img.size)
print(img.info)

img = img.resize((int(img.width*1.5),int(img.height*1.5)), resample=Image.LANCZOS)

tess_result = pytesseract.image_to_data(img,output_type='data.frame')#,config="--psm 3 --oem 1")
print("tess_result")
print(type(tess_result))
print(tess_result)
# get high confidence text
tess_result = tess_result[(tess_result['conf'] > 80)]
# get first block only
tess_result = tess_result[tess_result['block_num'] == 1]
# get text column
tess_result = tess_result['text']
print(tess_result)
# join each word
print(tess_result.str.cat(sep=' '))
