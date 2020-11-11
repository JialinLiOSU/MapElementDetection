import cv2
from shapely.geometry import Polygon
from shapely.geometry import box

path = r'C:\Users\jiali\Desktop\MapElementDetection\dataCollection\annotatedUSStates\lcc'
imgName = 'generated_annotated_USStates_lcc_5.png'
filename='generated_annotation_USStates_lcc.txt'
file = open(path + '\\' + filename,'r')
annotations = file.readlines()
annotationsCurrentImage = [anno for anno in annotations if anno.split(',')[0] == imgName]

img = cv2.imread(path + '\\'+imgName)

for anno in annotationsCurrentImage:
    annoElements = anno.split(',')
    coord = [float(annoElements[1]),float(annoElements[2]),float(annoElements[3]),float(annoElements[4])]
    width = float(annoElements[3])
    height = float(annoElements[4])
    if width * height > 100:
        startPoint = (int(coord[0]), int(coord[1]))
        endPoint = (int(coord[0]+coord[2]), int(coord[1]+coord[3]))
        cv2.rectangle(img,startPoint,endPoint,(255, 0, 0),2)
cv2.imshow(imgName, img)
cv2.waitKey(0)
cv2.destroyAllWindows()