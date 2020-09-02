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
    path = r"C:\Users\jiali\Desktop\PyTorch-YOLOv3\data\custom\labels"
    labelSourceDir = os.listdir(path)

    pathImages = r"C:\Users\jiali\Desktop\PyTorch-YOLOv3\data\custom\images"
    imageSourceDir = os.listdir(pathImages)
    imageFullNameList = []
    imageShortNameList = []
    # read image file names to a list ofr checking later
    for imagefile in imageSourceDir: 
        imageFullNameList.append(imagefile)
        imageShort = imagefile.split('.')[0]
        imageShortNameList.append(imageShort)
    width = 800
    height = 600

    pathTraValSource = r"C:\Users\jiali\Desktop\PyTorch-YOLOv3\data\custom"

    trainFileName = pathTraValSource + '\\' + 'train.txt'
    validFileName = pathTraValSource + '\\' + 'valid.txt'

    # read train files to a list for checking
    trainFiles = open(trainFileName, 'r')
    trainFileList = trainFiles.read().split('\n')
    trainFiles.close()
    trainNameShortList = []
    for trainFileLong in trainFileList:
        if trainFileLong == '' or trainFileLong == ' ':
            continue
        trainNameShort = trainFileLong.split('/')[3]
        trainNameShort = trainNameShort.split('.')[0]
        trainNameShortList.append(trainNameShort)

    # read valid files to a list for checking
    validFiles = open(validFileName, 'r')
    validFileList = validFiles.read().split('\n')
    validFiles.close()
    validNameShortList = []
    for validFileLong in validFileList:
        if validFileLong == '' or validFileLong == ' ':
            continue
        validNameShort = validFileLong.split('/')[3]
        validNameShort = validNameShort.split('.')[0]
        validNameShortList.append(validNameShort)

    trainList = []
    testList = []
    drivePathTrain = '/content/drive/My Drive/Map element detection/cocoFormatDataTrainTest/train/'
    drivePathTest = '/content/drive/My Drive/Map element detection/cocoFormatDataTrainTest/val/'

    # Opening JSON file 
    for labelfile in labelSourceDir: 
        # find the index of this image file, and get the image file name
        lfShort = labelfile.split('.')[0]
        idxImage = imageShortNameList.index(lfShort)
        imageFileName = imageFullNameList[idxImage]

        if lfShort in trainNameShortList:
            lfOpen = open(path + '\\' + labelfile, "r")
            lfData = lfOpen.read().split('\n')
            lfOpen.close()
            for singleLable in lfData:
                elementList = singleLable.split(' ')
                #elementList[12340] are centerX, centerY, labelWidth, labelHeight and labelID 
                if elementList[0] != '':
                    [xMin, xMax, yMin, yMax] = centerWHtoXYrange(float(elementList[1]), float(elementList[2]), float(elementList[3]), float(elementList[4]))
                    xMin = int(xMin * width)
                    xMax = int(xMax * width)
                    yMin = int(yMin * height)
                    yMax = int(yMax * height)
                    trainLineStr = drivePathTrain + imageFileName + ',' + str(xMin) + ',' + str(yMin) + ',' + \
                        str(xMax) + ',' + str(yMax) + ',' + elementList[0] + '\n'
                    trainList.append(trainLineStr)

        elif lfShort in validNameShortList:
            lfOpen = open(path + '\\' +  labelfile, "r")
            lfData = lfOpen.read().split('\n')
            lfOpen.close()
            for singleLable in lfData:
                elementList = singleLable.split(' ')
                if elementList[0] != '':
                    [xMin, xMax, yMin, yMax] = centerWHtoXYrange(float(elementList[1]), float(elementList[2]), float(elementList[3]), float(elementList[4]))
                    xMin = int(xMin * width)
                    xMax = int(xMax * width)
                    yMin = int(yMin * height)
                    yMax = int(yMax * height)
                    testLineStr = drivePathTrain + imageFileName + ',' + str(xMin) + ',' + str(yMin) + ',' + \
                        str(xMax) + ',' + str(yMax)  + ',' + elementList[0] + '\n'
                    testList.append(testLineStr)

    trainAnnName = pathTraValSource + '\\' + 'trainAnnoSelfDesign.txt'
    testAnnName = pathTraValSource + '\\' + 'testAnnoSelfDesign.txt'

    file = open(trainAnnName, 'a')
    file.writelines(trainList)
    file.close()

    file = open(testAnnName, 'a')
    file.writelines(testList)
    file.close()
    
        
if __name__ == "__main__":
    main()


    