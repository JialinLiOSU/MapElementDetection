# from titleResultsFinalGood, generate a txt for the list of titles
path = r'C:\Users\jiali\Desktop\MapElementDetection\code\postProcessingDetection'
filename='titleResultsFinalBad.txt'
file = open(path + '\\' + filename,'r')
titleResults = file.readlines()



titles = [nametitle.split(',')[1:]  for nametitle in titleResults]

titleList = []

for title in titles:
    if len(title) == 1:
        print(title[0])
    if len(title) > 1:
        titleTemp = ''
        for t in title:
            titleTemp = titleTemp + t
        print(titleTemp)
