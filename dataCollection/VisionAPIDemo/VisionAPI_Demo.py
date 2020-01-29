import os, io
from google.cloud import vision
from google.cloud.vision import types
import pandas as pd

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'ServiceAccountToken.json'

client = vision.ImageAnnotatorClient()
print (client)

FILE_NAME = 'imageLabel12.jpg'
FOLDER_PATH = r'C:\\Users\\li.7957\Desktop\\MapElementDetection\\dataCollection\\labeledMaps'

with io.open(os.path.join(FOLDER_PATH, FILE_NAME), 'rb') as image_file:
    content = image_file.read()

image = vision.types.Image(content=content)

response = client.text_detection(image= image)
df = pd.DataFrame(columns=['locale','description'])
print(response.text_annotations)