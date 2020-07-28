import cv2
import numpy as np
from matplotlib import pyplot as plt

path = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\map image segmentation\\sample images\\'

pathTemplateCA = 'templateCA.jpg'
pathTemplateCA1 = 'templateCA1.jpg'
pathTemplateCO = 'templateCO.jpg'
pathTemplateCO1 = 'templateCO1.jpg'
pathTemplateMI = 'templateMI.jpg'
pathTemplateTX = 'templateTX.jpg'
pathTemplateWI = 'templateWI.jpg'
pathTemplateWY = 'templateWY.jpg'
pathTemplateWY1 = 'templateWY1.jpg'
Image = 'ChoImg11.jpg'
img = cv2.imread(path+Image,0)
img2 = img.copy()
# dataPath = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\dataCollection\\labeledMapsWithCategory\\enhImages\\'
# # outputPath = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Legend Analysis\\legend images\\'
# imgName = 'ChoImg14.jpg'


# img = cv2.imread(dataPath + imgName)
templates = []

template = cv2.imread(path + pathTemplateWY,0)
templates.append(template)
template = cv2.imread(path + pathTemplateWY1,0)
templates.append(template)

w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = eval(meth)
    resList = []
    maxIndex = 0
    maxValue = -999999999
    for i in range(len(templates)):

        # Apply template Matching
        res = cv2.matchTemplate(img,templates[i],method)
        resList.append(res)
        if np.max(res)>maxValue:
            maxValue = np.max(res)
            maxIndex = i
    
    res = resList[maxIndex]
        
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img,top_left, bottom_right, 100, 2) # change rectangle color to dark gray

    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()