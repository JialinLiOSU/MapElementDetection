# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
# import cv2
from matplotlib import pyplot as plt
import pickle

def removeText(img, ocrResults):
    for ocr in ocrResults:
        bbox = ocr[0]
        xMin = int(bbox[0][0])
        yMin = int(bbox[0][1])
        xMax = int(bbox[2][0])
        yMax = int(bbox[2][1])
        value = img[yMax][xMax]

        for x in range(xMin,xMax):
            for y in range(yMin,yMax):
                img[y][x] = value

    return img

def main():
    # read detection results from pickle file
    detectResultName = r'C:\Users\jiali\Desktop\MapElementDetection\code\postProcessingDetection\detectResultsOrigin.pickle'
    with open(detectResultName, 'rb') as fDetectResults:
        detectResults = pickle.load(fDetectResults)

    # read ocr results from pickle file
    ocrResultName = r'C:\Users\jiali\Desktop\MapElementDetection\code\postProcessingDetection\ocrBoundsListOrigin.pickle'
    with open(ocrResultName, 'rb') as fOCRResults:
        ocrResults = pickle.load(fOCRResults)

    path = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\USStateChoro\originalSize'

    # img1Name = '1_i7aR525CtXLHOPbarZt5kg.png'
    # img1Name = '7.7886413.jpg'
    # img1Name = '22.Map.jpg'
    img1Name = '8.9781412956970-p406-1.jpg'
    # read images and remove texts on the images
    img1 = io.imread(path + '\\' + img1Name) # Image1 to be matched
    ocrImg1 = [ocr[1:] for ocr in ocrResults if ocr[0]==img1Name][0]
    img1Proc = removeText(img1,ocrImg1)

    image = img_as_float(img1Proc)
    # loop over the number of segments

    # apply SLIC and extract (approximately) the supplied number
    # of segments
    numSegments = 300
    segments = slic(image, n_segments = numSegments, sigma = 5)
    # show the output of SLIC
    fig = plt.figure("Superpixels -- %d segments" % (numSegments))
    ax = fig.add_subplot(1, 1, 1)
    bounds = mark_boundaries(image, segments)
    ax.imshow(bounds)
    plt.axis("off")
    # show the plots
    plt.show()

if __name__ == "__main__":    main()