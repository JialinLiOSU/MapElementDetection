# generate the label format according to PyTorch-YOLOv3
import json
import glob

path = r"C:\Users\jiali\Desktop\MapElementDetection\dataCollection\labeledMapsWithCategory\images"

# importing the module 

# Opening JSON file 
for filename in glob.glob(path+'/*.json'): 
    with open(filename) as json_file: 
        data = json.load(json_file) 
    
        # Print the type of data variable 
        print("Type:", type(data)) 
    