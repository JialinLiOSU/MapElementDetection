# generate the label format according to fit the self designed RCNN code
# The input is the labels in format for YOLOv3
# The output is the label annotations for both train and test set

import json
import glob
import os
import matplotlib.pyplot as plt
# import numpy as np
# np.random.seed(42)
# x = np.random.normal(size=1000)
# plt.hist(x, density=True, bins=30)  # `density=False` would make counts
# plt.ylabel('Probability')
# plt.xlabel('Data');

def main():
    path = r"C:\Users\jiali\Desktop\PyTorch-YOLOv3\data\custom\labels"
    labelSourceDir = os.listdir(path)
    ratioList = []

    for labelfile in labelSourceDir: 
        # find the index of this image file, and get the image file name
        lfShort = labelfile.split('.')[0]

        lfOpen = open(path + '\\' + labelfile, "r")
        lfData = lfOpen.read().split('\n')
        lfOpen.close()
        for singleLable in lfData:
            if singleLable == '' or singleLable == ' ':
                continue
            elementList = singleLable.split(' ')
            #elementList[12340] are centerX, centerY, labelWidth, labelHeight and labelID 
            labelWidth = float(elementList[3])
            labelHeight = float(elementList[4])
            ratio = labelWidth / labelHeight
            if ratio < 1:
                ratio = - 1/ratio
            ratioList.append(ratio)
    
    plt.hist(ratioList, density=True, bins=30)  # `density=False` would make counts
    plt.ylabel('Probability')
    plt.xlabel('width/height')
    plt.show()
    print('test')
           

        
if __name__ == "__main__":
    main()


    