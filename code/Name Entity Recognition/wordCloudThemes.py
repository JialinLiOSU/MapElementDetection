# Python program to generate WordCloud 
  
# importing all necessery modules 
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt 
import pandas as pd 
  
# read title data
themePath = r'C:\Users\jiali\Desktop\MapElementDetection\code\Name Entity Recognition'
themeFileName = 'themeResultsCleanup.txt'

f = open(themePath + '\\' + themeFileName, "r")
themeFile = f.read()
themePairs = themeFile.split('\n')

themeList = []
for tp in themePairs:
    tpElements = tp.split(',')
    if len(tpElements)>3:
        theme = tpElements[-2]
        themeList.append(theme.lower())

#convert it to dictionary with values and its occurences
from collections import Counter
word_could_dict=Counter(themeList)
wordcloud = WordCloud(width = 1000, height = 600,min_font_size = 10).generate_from_frequencies(word_could_dict)

plt.figure(figsize=(10,8),dpi=150)
plt.imshow(wordcloud)
plt.axis("off")
# plt.show()
plt.savefig('themeWordCloud.png', bbox_inches='tight')
plt.close()