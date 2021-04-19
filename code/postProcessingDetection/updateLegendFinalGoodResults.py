import pickle
import os
    
    
    
# legendResults.append((imgName,finalLegendBox,legendRectShapeBoxList,legendTextShapelyBoxList,legendTextBboxes))
with open(r'C:\Users\jiali\Desktop\MapElementDetection\code\postProcessingDetection\legendFinalGoodResults_1.pickle', 'rb') as f:
	legendResults_1 = pickle.load(f)

# read results from colorsMappingAreaResults to get mapping area colors
# colorsMappingAreaResults.append((img1Name, superPixelValueList,superPixelValueList))
with open(r'C:\Users\jiali\Desktop\MapElementDetection\code\postProcessingDetection\legendFinalGoodResults.pickle', 'rb') as f:
	legendResults = pickle.load(f)

imgName = '115936742_3317778874932504_6087693444288850821_o.jpg'

newLegendResults = []
for lr1 in legendResults_1:
    if lr1[0] == imgName:
        finalLegendBoxTemp = lr1[1]
        legendTextShapelyBoxListTemp = lr1[3]
        legendTextBboxesTemp = lr1[4]

for lr in legendResults:
    if lr[0] != imgName:
        newLegendResults.append(lr)
    else:
        newLegendResults.append((lr[0],finalLegendBoxTemp,lr[2],legendTextShapelyBoxListTemp,legendTextBboxesTemp))




with open(r'C:\Users\jiali\Desktop\MapElementDetection\code\postProcessingDetection\legendFinalGoodResultsNew.pickle', 'wb') as f:
	pickle.dump(newLegendResults,f)