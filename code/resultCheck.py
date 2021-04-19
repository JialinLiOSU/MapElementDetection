import pickle

path = r'C:\Users\jiali\Desktop\MapElementDetection\code\Legend Analysis'
fileName = 'legendMatchingResultsFinalBad.pickle'

with open(path + '\\' + fileName, 'rb') as f:
    legendMatchingResults = pickle.load(f)

print('test')