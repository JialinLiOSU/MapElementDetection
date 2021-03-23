import sys
import os

path = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\USStateChoro\finalTest'
testImageDir = os.listdir(path)
for img in testImageDir:
    if img[-4:] == 'json':
        continue
    print(img)