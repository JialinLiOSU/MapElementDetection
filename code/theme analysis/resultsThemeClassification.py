import pickle
import random
import csv

def id2type(themeID):
    if themeID == 0:
        return 'other'
    elif themeID == 1:
        return 'demographic'
    elif themeID == 2:
        return 'economic'
    elif themeID == 3:
        return 'physical'
    else:
        return 'social'

def main():
    with open('C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\theme analysis\\test_theme.pkl', 'rb') as f:
        testThemes = pickle.load(f)
    with open('C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\theme analysis\\test_pred_label.pkl', 'rb') as f:
        testPredLabel = pickle.load(f)

    themes = testThemes._mgr._values

    # with open('C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\theme analysis\\testThemePred.txt', 'w') as f:
    rows = []
    for i in range(len(testPredLabel)):
        rows.append([themes[i], id2type(testPredLabel[i])])
        # f.write(themes[i]  + ',    ' + id2type(testPredLabel[i]) + '\n')
    
    # name of csv file  
    path = r'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\theme analysis'
    filename = "testThemePred.csv"

    fields = ['theme','label']
        
    # writing to csv file  
    with open(path + '\\' + filename, 'w') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  
        # writing the fields  
        csvwriter.writerow(fields)  
        # writing the data rows  
        csvwriter.writerows(rows) 

    print('test')


if __name__ == "__main__":
    main()
