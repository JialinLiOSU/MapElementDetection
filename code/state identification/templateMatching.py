import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle

# find the dominant color value for each legend rectangle
def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]
def rgb2Grey(dominantColor):
    rgb_weights = [0.2989, 0.5870, 0.1140]
    dominantColorGrey = int(np.dot(dominantColor, rgb_weights))
    return dominantColorGrey

def main():

    path = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\state identification\\sample images\\'
    validStateNames = ['Alabama','Arkansas','Arizona','California',
        'Colorado','Connecticut','Delaware','Florida','Georgia','Iowa','Idaho','Illinois',
        'Indiana','Kansas','Kentucky','Louisiana','Massachusetts','Maryland','Maine','Michigan','Minnesota','Missouri',
        'Mississippi','Montana','North Carolina','North Dakota','Nebraska','New Hampshire','New Jersey',
        'New Mexico','Nevada','New York','Ohio', 'Oklahoma','Oregon','Pennsylvania','Rhode Island','South Carolina',
        'South Dakota','Tennessee','Texas','Utah','Virginia','Vermont','Washington','Wisconsin','West Virginia','Wyoming']

    templatesPathRoot = path + 'templates\\'
    templatePathList = [templatesPathRoot + state for state in validStateNames]

    templatesStates = []
    for tempOneStatePath in templatePathList:
        templatesOneState = []
        for i in range(2):
            temp = os.listdir(tempOneStatePath)[i]
            template = cv2.imread(tempOneStatePath +'\\'+ temp,0)
            templatesOneState.append(template)
        templatesStates.append(templatesOneState)

    # All the 6 methods for comparison in a list
    methods = [
                'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    indexMethod = 5
    meth = methods[indexMethod]

    testPath = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\annotatedUSStates\template matching testing\images'
    testImageNames = os.listdir(testPath)
    resultsImagesList = []
    for Image in testImageNames:
        # Image = 'USA_states_population_density_map.png'
        img = cv2.imread(testPath + '\\' + Image,0)
        img2 = img.copy()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # w, h = template.shape[::-1]
        greyStates = []
        colorStates = []
        for templates in templatesStates: # templates is for one state
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
            center = (int(top_left[0] + w/2),int(top_left[1] + h/2) )

            centerColor = imgRGB[center[1],center[0]]
            centerGrey = rgb2Grey(centerColor)
            
            greyStates.append(centerGrey)
            colorStates.append(centerColor)
        resultsImagesList.append((greyStates,colorStates))
    with open(r'C:\Users\jiali\Desktop\MapElementDetection\code\state identification\templateResults' + str(indexMethod) + '.pickle', 'wb') as f:
	    pickle.dump(resultsImagesList,f)

if __name__ == "__main__":    main()


            # croppedImg = imgRGB[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
            # dominantColor = unique_count_app(croppedImg)
            # dominantColorGrey = rgb2Grey(dominantColor)

            # cv2.rectangle(img,top_left, bottom_right, 100, 2) # change rectangle color to dark gray

            # plt.subplot(121),plt.imshow(res,cmap = 'gray')
            # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            # plt.subplot(122),plt.imshow(img,cmap = 'gray')
            # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            # plt.suptitle(meth)

            # plt.show()