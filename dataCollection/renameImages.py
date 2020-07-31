# preprocess the images
from PIL import Image
import glob

pathSource='C:\\Users\\jiali\\Desktop\\MapElementDetection\\dataCollection\\new choropleth map'
pathTarget='C:\\Users\\jiali\\Desktop\\MapElementDetection\\dataCollection\\preprocessedMaps\\'

width=800
height=600
i=904
for filename in glob.glob(pathSource+'/*.jpeg'): 
    img=Image.open(filename)
    img = img.resize((width, height), Image.ANTIALIAS)
    # if img.mode != 'RGB':
    name_target='imageLabel'+str(i+1)+'.jpeg'
    img.save(pathTarget+name_target)
    i=i+1