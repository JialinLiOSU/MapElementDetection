### Image preprocess before NN training ###
from PIL import Image
import numpy as np
import os
from PIL import ImageChops

class MapImage:
    def __init__(self, name, image, isExist = False):
        self.name = name
        self.image = image
        self.isExist = isExist

def hasSameImage(oMapImage,cMapImageList):
    for cMapImage in cMapImageList:
        if oMapImage.image.shape == cMapImage.image.shape and (oMapImage.image == cMapImage.image).all():
            return True,cMapImage.name
    return False, None
    pass

def main():

    pathComplete=r'C:\Users\jiali\Desktop\PyTorch-YOLOv3\data\custom\images'
    pathOrigin=r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\original size choro maps'
    completeMapImages = os.listdir(pathComplete)
    originMapImages = os.listdir(pathOrigin)
    cMapImageList = []
    for cmapImage in completeMapImages:
        nameSource=pathComplete + '\\' + cmapImage
        img = Image.open(nameSource)
        img = np.array(img) 
        mapImage = MapImage(cmapImage, img)
        cMapImageList.append(mapImage)
    oMapImageList = []
    width, height = 800, 600
    for omapImage in originMapImages:
        nameSource=pathOrigin + '\\' + omapImage
        img = Image.open(nameSource)
        img = img.resize((width, height))
        img = np.array(img) 
        mapImage = MapImage(omapImage, img)
        oMapImageList.append(mapImage)
        # img = Image.open(path_source+name_source).LA
    oMapImageListProc = []
    for cMapImage in cMapImageList:
        isSame,oMapImageName = hasSameImage(cMapImage,oMapImageList)
        if isSame:
            cMapImage.isExist = True
            print('exist:'+cMapImage.name+ ','+ oMapImageName)
            oMapImageListProc.append(cMapImage)
        else:
            print('not exist:'+cMapImage.name)
    
    
if __name__ == "__main__":
    main()
    pass
# img.show()
# print(img)

