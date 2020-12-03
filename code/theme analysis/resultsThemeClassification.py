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

    with open('C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\theme analysis\\testThemePred.txt', 'w') as f:
        for i in range(len(testPredLabel)):
            f.write(themes[i]  + ',    ' + id2type(testPredLabel[i]) + '\n')

    print('test')


if __name__ == "__main__":
    main()
