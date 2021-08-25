# in our study, the annotation tool used is labelme, which is good and stable to use
# but the format of annotation data to be used for R-CNN based detection model is coco format
# therefore, we should convert the labelme data to coco data

# import package
import labelme2coco

# set directory that contains labelme annotations and image files
labelme_folder = r"C:\Users\jiali\Desktop\thematicMapElementDetection\labeledThematicMaps"

# set path for coco json to be saved
save_json_path = r"C:\Users\jiali\Desktop\thematicMapElementDetection" + '\\' + "cocoThematicMaps.json"
# train_finaltest_bad.json

# convert labelme annotations to coco
labelme2coco.convert(labelme_folder, save_json_path)
# importing the module
