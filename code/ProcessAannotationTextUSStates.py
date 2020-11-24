# process the annotations for US state maps 
# to make sure there is only one rectangle for each state in continental US
import cv2
from shapely.geometry import Polygon
from shapely.geometry import box
from math import ceil

path = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\annotatedUSStates'
ceaPath = path + '\\lcc'
# '\\generated_annotated_USStates_lcc_10.png'
filename='generated_annotation_USStates_lcc.txt'
file = open(path + '\\' + filename,'r')
annotations = file.readlines()

# us state name and acronym
stateNames = ['Alabama','Arkansas','Arizona','California',
    'Colorado','Connecticut','District of Columbia','Delaware','Florida','Georgia','Iowa','Idaho','Illinois',
    'Indiana','Kansas','Kentucky','Louisiana','Massachusetts','Maryland','Maine','Michigan','Minnesota','Missouri',
    'Mississippi','Montana','North Carolina','North Dakota','Nebraska','New Hampshire','New Jersey',
    'New Mexico','Nevada','New York','Ohio', 'Oklahoma','Oregon','Pennsylvania','Rhode Island','South Carolina',
    'South Dakota','Tennessee','Texas','Utah','Virginia','Vermont','Washington','Wisconsin','West Virginia','Wyoming']

imageNameList = ['generated_annotated_USStates_lcc_' + str(i) + '.png' for i in range(400)]
strList = []
statePositions = []
imageName = imageNameList[0]
annotationsCurrentImage = [anno for anno in annotations if anno.split(',')[0] == imageName]

for state in stateNames:
    annotationsCurrentState = [anno for anno in annotationsCurrentImage if anno.split(',')[-1].startswith(state)]
    area = 0
    maxIndex = 0

    for i in range(len(annotationsCurrentState)):
        annoElements = annotationsCurrentState[i].split(',')
        width = float(annoElements[3])
        height = float(annoElements[4])
        if width * height > area:
            area = height * width
            maxIndex = i
    
    statePosition = annotationsCurrentState[maxIndex].split(',')[1:]
    statePosition = str(int(float(statePosition[0]))) + ',' + str(int(float(statePosition[1]))) + ',' \
                        + str(int(ceil(float(statePosition[2])))) + ',' + str(int(ceil(float(statePosition[3])))) + ',' + statePosition[4]
    statePositions.append(statePosition)

for imageName in imageNameList:
    for sp in statePositions:
        strLine = imageName + ',' + sp
        strList.append(strLine)

filename='Processed_annotation_USStates_lcc'+'.txt'
file = open(path +'\\'+ filename,'a')
file.writelines(strList)
                
