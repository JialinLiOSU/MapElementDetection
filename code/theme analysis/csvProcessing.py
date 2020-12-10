import csv

path = r'C:\Users\jiali\Desktop\MapElementDetection\code\theme analysis'
fileName = 'realCorrectLabelThemes.csv'
csvLines = []
with open(path + '\\' + fileName) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if row[0] != '':
            csvLines.append(row)

filename = "realCorrectLabelThemesProced.csv"
# writing to csv file  
with open(path + '\\' + filename, 'w') as csvfile:  
    csvwriter = csv.writer(csvfile)  
    csvwriter.writerows(csvLines) 