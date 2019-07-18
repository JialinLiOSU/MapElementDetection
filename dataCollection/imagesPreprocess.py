# preprocess the images
from PIL import Image
import glob

pathSource='C:\\Users\\li.7957\\Desktop\\MapElementDetection\\dataCollection\\choropleth map'
pathTarget='C:\\Users\\li.7957\\Desktop\\MapElementDetection\\dataCollection\\preprocessedMaps\\'

width=800
height=600
i=0
for filename in glob.glob(pathSource+'/*.jpg'): #assuming gif
    img=Image.open(filename)
    img = img.resize((width, height), Image.ANTIALIAS)
    
    name_target='imageLabel'+str(i+1)+'.jpg'
    img.save(pathTarget+name_target)
    i=i+1