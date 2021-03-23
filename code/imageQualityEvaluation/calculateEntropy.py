# The goal is to identify US states 
# test achieve this goal in two steps
# 1. edge detection for the original images
# 2. feature matching by SIFT descriptor
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import skimage.measure


import numpy as np
import cv2
from matplotlib import pyplot as plt
import pickle
import numpy as np
import pickle
import os
import sys
import random



def main():

    testImagePath = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\USStateChoro\legendSamples'
    testImageDir = os.listdir(testImagePath)
    
    for img1Name in testImageDir:
        print('image name: ' + img1Name)
        img1 = skimage.io.imread(testImagePath + '\\' +img1Name, True)
        # img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        entropy = skimage.measure.shannon_entropy(img1)
        
        fig = plt.figure()
        ax = plt.gca()
        ax.imshow(img1)
        plt.title(img1Name + ': '+str(entropy)) 
        # plt.show()
        fig.savefig(testImagePath + '\\entropy_' +img1Name)
    
if __name__ == "__main__":    main()