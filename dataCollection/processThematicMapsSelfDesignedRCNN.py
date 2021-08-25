# generate the label format according to fit the self designed RCNN code
# The input is the labels in format for YOLOv3
# The output is the label annotations for both train and test set

import json
import glob
import os

def centerWHtoXYrange(centerX, centerY, width, height):
    xMin = centerX - width/2
    xMax = centerX + width/2
    yMin = centerY - height/2
    yMax = centerY + height/2
    return [xMin, xMax, yMin, yMax]

def main():
    json_folder = r"C:\Users\jiali\Desktop\thematicMapElementDetection"

    thematicMapCocoPath = json_folder + '\\' + 'cocoThematicMaps.json'

    drivePath = '/content/drive/My Drive/Map_element_detection/cocoFormatDataTrainTest/'
    
    with open(thematicMapCocoPath) as json_file: 
        thematicMapJson = json.load(json_file) 

    # process label data for final test good
    goodStrList = []
    for anno in thematicMapJson['annotations']:
        image_id = anno['image_id']
        image_name_path = thematicMapJson['images'][image_id - 1]['file_name']
        image_name = image_name_path.split('\\')[-1]
        bbox = anno['bbox']
        xMin = bbox[0]
        yMin = bbox[1]
        xMax = xMin + bbox[2]
        yMax = yMin + bbox[3]
        xMin = int(xMin)
        xMax = int(xMax)
        yMin = int(yMin)
        yMax = int(yMax)
        if anno['category_id'] == 1:
            label_id = 0
        elif anno['category_id'] == 2:
            label_id = 1
        else:
            continue
        goodLineStr = drivePath + 'train/' + image_name + ',' + str(xMin) + ',' + str(yMin) + ',' + \
                        str(xMax) + ',' + str(yMax) + ',' +str(label_id) + '\n'
        goodStrList.append(goodLineStr)

    finalGoodAnnName = json_folder + '\\' + 'thematicMapsAnnoSelfDesign.txt'
    file = open(finalGoodAnnName, 'a',encoding='utf-8')
    file.writelines(goodStrList)
    file.close()
        
if __name__ == "__main__":
    main()


    