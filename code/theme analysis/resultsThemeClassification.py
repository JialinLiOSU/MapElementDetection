import pickle
import random
import csv

def id2type(themeID):
    if themeID == 0:
        return 'other'
    elif themeID == 1:
        return 'social'
    elif themeID == 2:
        return 'economic'
    else:
        return 'environmental'


def main():
    with open('C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\theme analysis\\test_theme.pkl', 'rb') as f:
        testThemes = pickle.load(f)
    with open('C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\theme analysis\\test_pred_label.pkl', 'rb') as f:
        testPredLabel = pickle.load(f)
    with open('C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\theme analysis\\pred_prob.pkl', 'rb') as f:
        testPredProb = pickle.load(f)

    themes = testThemes._mgr._values

    # with open('C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\theme analysis\\testThemePred.txt', 'w') as f:
    rows = []
    for i in range(len(testPredLabel)):
        rows.append([themes[i], id2type(testPredLabel[i])])
        # f.write(themes[i]  + ',    ' + id2type(testPredLabel[i]) + '\n')
    
    # name of csv file  
    path = r'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\theme analysis'
    filename = "themeClassificationResults.csv"

    fields = ['theme','label']
        
    # writing to csv file  
    with open(path + '\\' + filename, 'w') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)    
        # writing the fields  
        for i in range(len(rows)):
            csvwriter.writerow(rows[i])
            csvwriter.writerow(testPredProb[i])
        # csvwriter.writerow(testPredProb[i])  
        # # writing the data rows  
        # csvwriter.writerows(rows) 

    print('test')


if __name__ == "__main__":
    main()
