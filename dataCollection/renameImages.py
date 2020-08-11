# preprocess the images
from PIL import Image
import glob

pathSource='C:\\Users\\jiali\\Desktop\\MapElementDetection\\dataCollection\\lasted add choropleth maps'
pathTarget='C:\\Users\\jiali\\Desktop\\MapElementDetection\\dataCollection\\preprocessedMaps\\'

width=800
height=600
i=944
for filename in glob.glob(pathSource+'/*.jpg'): 
    img=Image.open(filename)
    img = img.resize((width, height), Image.ANTIALIAS)
    # if img.mode != 'RGB':
    name_target='ChoImg'+str(i+1)+'.jpg'
    img.save(pathTarget+name_target)
    i=i+1