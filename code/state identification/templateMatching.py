import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

path = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\state identification\\sample images\\'
templatesPath = path + 'templates\\'
templatesPathAL = templatesPath + 'AL\\'
templatesPathHA = templatesPath + 'HA\\'

templates = []
for temp in os.listdir(templatesPathAL):
    template = cv2.imread(templatesPathAL + temp,0)
    templates.append(template)


testPath = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\USStateChoro\ContigUSBoundFinalGood'
testImageNames = os.listdir(testPath)
for Image in testImageNames:
    Image = 'USA_states_population_density_map.png'
    img = cv2.imread(path + Image,0)
    img2 = img.copy()
    # w, h = template.shape[::-1]

    # All the 6 methods for comparison in a list
    methods = [
                'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    for meth in methods:
        img = img2.copy()
        method = eval(meth)
        resList = []
        maxIndex = 0
        maxValue = -999999999
        for i in range(0,len(templates)):

            # Apply template Matching
            if i < 2 and templates[i] is not None:
                res = cv2.matchTemplate(img,templates[i],method)
                resList.append(res)
                if np.max(res)>maxValue:
                    maxValue = np.max(res)
                    maxIndex = i
        
        res = resList[maxIndex]
        template =templates[maxIndex]
        w, h = template.shape[::-1]
            
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