# preprocess the images
from PIL import Image
import glob
import os
from shutil import copyfile


pathNotExist=r'C:\Users\jiali\Desktop\PyTorch-YOLOv3\data\custom\notExistImages'
pathOriginExist=r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\originExistImages'
pathOriginNotExist=r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\originNotExistImages'

pathComplete=r'C:\Users\jiali\Desktop\PyTorch-YOLOv3\data\custom\images'
pathOrigin=r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\original size choro maps'
originMapImages = os.listdir(pathOrigin)

pathTxt = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection'
nameTxt = 'compareResults.txt'

fileCompareResults = open(pathTxt + '\\' + nameTxt,"r+")  
compareResults = fileCompareResults.readlines() 

notExistList = []
originExistList = []
originNotExistList = []
for cr in compareResults:
    if cr.split(':')[0] == 'not exist':
        nameWithN = cr.split(':')[1]
        if nameWithN[-1] == '\n':
            name = nameWithN[:-1]
        else:
            name = nameWithN
        
        notExistList.append(name)
    elif cr.split(':')[0] == 'exist':
        nameWithN = cr.split(':')[1].split(',')[1]
        if nameWithN[-1] == '\n':
            name = nameWithN[:-1]
        else:
            name = nameWithN
        originExistList.append(name)

for omi in originMapImages:
    if omi not in originExistList:
        originNotExistList.append(omi)

print('test')
# for ne in notExistList:
#     img=Image.open(pathComplete + '\\' + ne)
#     img.save(pathNotExist + '\\' + ne)

for oe in originExistList:
    copyfile(pathOrigin + '\\' + oe, pathOriginExist + '\\' + oe)

for one in originNotExistList:
    copyfile(pathOrigin + '\\' + one, pathOriginNotExist + '\\' + one)
