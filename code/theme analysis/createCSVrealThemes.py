
import random
import csv
import pickle


def main():
    with open('C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Name Entity Recognition\\theme_embedding_BERT\\themeList.pkl', 'rb') as f:
        themeList = pickle.load(f)

    labelList = []
    for i in range(len(themeList)):
            labelList.append('0')

    # field names  
    fields = ['label', 'theme']  
    rows = []
    for i in range(len(themeList)):
        rows.append([labelList[i], themeList[i]])
        
    # name of csv file  
    path = r'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\theme analysis'
    filename = "realWrongLabelThemes.csv"
        
    # writing to csv file  
    with open(path + '\\' + filename, 'w') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  
        # writing the fields  
        csvwriter.writerow(fields)  
        # writing the data rows  
        csvwriter.writerows(rows) 


if __name__ == "__main__":
    main()
