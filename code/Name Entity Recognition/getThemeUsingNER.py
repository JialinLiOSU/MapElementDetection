from __future__ import unicode_literals, print_function

import random
import numpy as np
import os
from spacy.util import minibatch, compounding
import spacy
from pathlib import Path
import plac
import en_core_web_sm
from spacy.gold import GoldParse
from spacy.scorer import Scorer

# new entity label
theme = "THEME"
region = "GPE"
time = "DATE"
admin = "ADMIN"

import pickle
# nlpTest = en_core_web_sm.load()

# read title data
titlePath = r'C:\Users\jiali\Desktop\MapElementDetection\code\postProcessingDetection'
titleFileName = 'titleResultsFinalGood.txt'

f = open(titlePath + '\\' + titleFileName, "r")
titleFile = f.read()
titlePairs = titleFile.split('\n')
if titlePairs[-1] ==None:
    titlePairs = titlePairs[0:-1]

imageNameList = []
titleList = []
pairList = []
for titlePair in titlePairs:
    imageName = titlePair.split(',')[0]
    imageNameList.append(imageName)
    imgNameLength = len(imageName)
    title = titlePair[imgNameLength+1:]
    if len(title) != 0 and title[-1] == '_':
        title = title[0:-2]
    if len(title) != 0:
        titleList.append(title)
        pairList.append([imageName,title])


strList = []
# save model to output directory
modelPath = r'D:\OneDrive - The Ohio State University\Map understanding\NER'
with (open(modelPath + '\\' + 'en_core_web_sm_THEME.pkl', "rb")) as openfile:
    nlp = pickle.load(openfile)

for pair in pairList:
    pair[1] = pair[1].replace("_", " ")
    doc = nlp(pair[1])
    print("Entities in '%s'" % pair[1])
    strtemp = pair[0] + '@' + pair[1] + '@' 
    for ent in doc.ents:
        print(ent.label_, ent.text)
        if ent.label_ == 'THEME':
            strtemp = strtemp + ent.text + '@'
    strtemp = strtemp + '\n'
    strList.append(strtemp)

file = open(r'C:\Users\jiali\Desktop\MapElementDetection\code\Name Entity Recognition\themeResultsFinalGoodTest.txt','a')
file.writelines(strList)

print('test')

# test the saved model
# print("Loading from", output_dir)
# nlp2 = spacy.load(output_dir)
# # Check the classes have loaded back consistently
# assert nlp2.get_pipe("ner").move_names == move_names
# doc2 = nlp2(test_text)
# for ent in doc2.ents:
#     print(ent.label_, ent.text)

