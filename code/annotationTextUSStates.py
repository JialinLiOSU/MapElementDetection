import cv2
from shapely.geometry import Polygon
from shapely.geometry import box

imagePath = r'C:\Users\jiali\Desktop\MapElementDetection\code\shpFiles'
imgName = 'generated_legend_CA_0.png'
img = cv2.imread(imagePath + '\\'+imgName)
coord = [536.3483638275379,148.14067725344177,72.61569976272733,37.310737207020935]
startPoint = (int(coord[0]), int(coord[1]))
endPoint = (int(coord[0]+coord[2]), int(coord[1]+coord[3]))

cv2.rectangle(img,startPoint,endPoint,(255, 0, 0),2)
cv2.imshow(imgName, img)
cv2.waitKey(0)
cv2.destroyAllWindows()