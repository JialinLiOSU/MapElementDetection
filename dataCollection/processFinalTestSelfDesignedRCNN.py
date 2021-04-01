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
    json_folder = r"C:\Users\jiali\Desktop\MapElementDetection\dataCollection\USStateChoro"

    finalTestGoodPath = json_folder + '\\' + 'train_finaltest_good.json'
    finalTestBadPath = json_folder + '\\' + 'train_finaltest_bad.json'
    drivePath = '/content/drive/My Drive/Map_element_detection/cocoFormatDataTrainTest/'
    
    with open(finalTestGoodPath) as json_file: 
        finalTestGoodJson = json.load(json_file) 
    with open(finalTestBadPath) as json_file: 
        finalTestBadJson = json.load(json_file) 

    # process label data for final test good
    goodStrList = []
    for anno in finalTestGoodJson['annotations']:
        image_id = anno['image_id']
        image_name_path = finalTestGoodJson['images'][image_id - 1]['file_name']
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
            label_id = 1
        else:
            label_id = 0
        goodLineStr = drivePath + 'finalGood/' + image_name + ',' + str(xMin) + ',' + str(yMin) + ',' + \
                        str(xMax) + ',' + str(yMax) + ',' +str(label_id) + '\n'
        goodStrList.append(goodLineStr)

    finalGoodAnnName = json_folder + '\\' + 'finalGoodTestAnnoSelfDesign.txt'
    file = open(finalGoodAnnName, 'a')
    file.writelines(goodStrList)
    file.close()

    # process label data for final test bad
    badStrList = []
    for anno in finalTestBadJson['annotations']:
        image_id = anno['image_id']
        image_name_path = finalTestBadJson['images'][image_id - 1]['file_name']
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
            label_id = 1
        else:
            label_id = 0
        badLineStr = drivePath + 'finalBad/' + image_name + ',' + str(xMin) + ',' + str(yMin) + ',' + \
                        str(xMax) + ',' + str(yMax) + ',' +str(label_id) + '\n'
        badStrList.append(badLineStr)

    finalBadAnnName = json_folder + '\\' + 'finalBadTestAnnoSelfDesign.txt'
    file = open(finalBadAnnName, 'a')
    file.writelines(badStrList)
    file.close()
        
if __name__ == "__main__":
    main()


    