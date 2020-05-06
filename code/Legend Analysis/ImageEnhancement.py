# this file is used to enhance the legend areas which are from labeled images
# The enhanced images are easier to conduct further analysis
# Author: Jialin Li
# Date: 5/24/2020
from PIL import Image
from PIL import ImageEnhance
import os

dataPath = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Legend Analysis\\legend images\\'
outputPath = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Legend Analysis\\enhanced legend images\\'

def imageEnhancement(image, propToBeEnhanced):
    # parameter propToBeEnhanced is a int from 1 to 4
    # representing brightness, color, contrast and sharpness respectively

    # enhance on brightness
    if propToBeEnhanced == 1:
        enh_bri = ImageEnhance.Brightness(image)
        brightness = 1.5
        image_enhanced = enh_bri.enhance(brightness)
        # image_brightened.show()
    elif propToBeEnhanced == 2:
        # enhance on color
        enh_col = ImageEnhance.Color(image)
        color = 1.5
        image_enhanced = enh_col.enhance(color)
        # image_colored.show()
    elif propToBeEnhanced == 3:
        # enhance on contrast
        enh_con = ImageEnhance.Contrast(image)
        contrast = 1.5
        image_enhanced = enh_con.enhance(contrast)
        # image_contrasted.show()
    elif propToBeEnhanced == 4:
        # enhance on sharpness
        enh_sha = ImageEnhance.Sharpness(image)
        sharpness = 3.0
        image_enhanced = enh_sha.enhance(sharpness)
        # image_sharped.show()
    else:
        print('propToBeEnhanced should be from 1 to 4')
    return image_enhanced


def main():
    for filename in os.listdir(dataPath):
        img = Image.open(dataPath + "\\"+filename)
        enhImg = imageEnhancement(img, 3)
        enhImg = imageEnhancement(img, 4)
        enhImg.save(outputPath + "\\" + filename)


if __name__ == "__main__":
    main()

