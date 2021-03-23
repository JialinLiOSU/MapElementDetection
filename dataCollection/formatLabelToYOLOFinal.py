# generate the label format according to PyTorch-YOLOv3
import json
import glob

path = r"C:\Users\jiali\Desktop\MapElementDetection\dataCollection\USStateChoro\finalTestBad"

# importing the module 

# Opening JSON file 
for filename in glob.glob(path+'/*.json'): 
    labelList = []
    with open(filename) as json_file: 
        data = json.load(json_file) 
        height = data['imageHeight']
        width = data['imageWidth']

        for shape in data['shapes']:
            maxX = 0
            maxY = 0
            minX = width
            minY = height

            if shape['label'] == "title":
                labelID = 0
            elif shape['label'] =="legend":
                labelID = 1
            else:
                print("filename: " + filename)

            for point in shape['points']:
                if point[0] > maxX:
                    maxX = point[0]
                if point[0] < minX:
                    minX = point[0]
                if point[1] > maxY:
                    maxY = point[1]
                if point[1] < minY:
                    minY = point[1]
            labelHeight = maxY - minY
            labelWidth = maxX - minX
            centerX = (maxX + minX)/2
            centerY = (maxY + minY)/2

            centerX = centerX / width
            centerY = centerY / height
            labelWidth = labelWidth / width
            labelHeight = labelHeight / height

            labelTemp = str(labelID) + ' ' + str(centerX) + ' ' + str(centerY) + ' ' + str(labelWidth) + ' ' + str(labelHeight) + ' ' +'\n'
            labelList.append(labelTemp)
    filename = filename.split('.')[0]
    filename = filename + '.txt'
    file = open(filename, 'a')
    file.writelines(labelList)
    # file.writelines(incorrectImgNameStrList)
    file.close()
    
        


    