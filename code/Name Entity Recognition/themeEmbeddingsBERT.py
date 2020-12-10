# calculate BERT embeddings for the themes
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# from unidecode import unidecode
import string
nltk.download('stopwords')
from collections import Counter
import itertools
import pickle
import math
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import numpy as np
import gensim.downloader as api
wv = api.load('word2vec-google-news-300')

def map_word_frequency(document):
    return Counter(document)

def get_weighted_feature_vectors(themeEmbeddingDict, word_counts, word_emb_model=wv):
    # sentence = [token for token in sentence1.split() if token in word_emb_model.vocab]
    embedding_size = len(themeEmbeddingDict['[CLS]']) # size of vectore in word embeddings
    a = 0.001
   
    vs = np.zeros(embedding_size)
    sentence_length = len(themeEmbeddingDict) - 1
    keys = themeEmbeddingDict.keys()
    for key in keys:
        a_value = a / (a + word_counts[key]) # smooth inverse frequency, SIF
        vs = np.add(vs, np.multiply(a_value, themeEmbeddingDict[key])) # vs += sif * word_vector
    vs = np.divide(vs, sentence_length) # weighted average
    return vs

def get_average_feature_vectors(themeEmbeddingDict):
    embedding_size = len(themeEmbeddingDict['[CLS]'])
    numToken = len(themeEmbeddingDict) - 1
    vs = np.zeros(embedding_size)
    # sentence_length = len(sentence1)
    for emdKey in themeEmbeddingDict.keys():
        if emdKey == '[CLS]':
            continue
        vs = np.add(vs, themeEmbeddingDict[emdKey]) 
    vs = np.divide(vs, numToken) # weighted average
    return vs

# def pre_process(corpus):
#     # convert input corpus to lower case.
#     corpus = corpus.lower()
#     # collecting a list of stop words from nltk and punctuation form
#     # string class and create single array.
#     stopset = stopwords.words('english') + list(string.punctuation)
#     # remove stop words and punctuations from string.
#     # word_tokenize is used to tokenize the input corpus in word tokens.
#     corpus = " ".join([i for i in word_tokenize(corpus) if i not in stopset])
#     # remove non-ascii characters
#     # corpus = unidecode(corpus)
#     return corpus
def get_cosine_similarity(feature_vec_1, feature_vec_2):    
    return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]

def main():

    themePath = r'C:\Users\jiali\Desktop\MapElementDetection\code\Name Entity Recognition\theme_embedding_BERT'
    # read theme list for labeling in dendrogram
    themeFileName = 'SimplifiedthemeList.pkl'
    with open(themePath + '\\' + themeFileName, 'rb') as f:
        themeList = pickle.load(f)

    # # read theme embeddings from BERT model
    # themeEmbeddingFileName = 'theme_BERT_embeddings_20.pkl'
    # with open(themePath + '\\' + themeEmbeddingFileName, 'rb') as f:
    #     themeEmbeddings1 = pickle.load(f)
    # themeEmbeddingFileName = 'theme_BERT_embeddings_30.pkl'
    # with open(themePath + '\\' + themeEmbeddingFileName, 'rb') as f:
    #     themeEmbeddings2 = pickle.load(f)
    # themeEmbeddingFileName = 'theme_BERT_embeddings_200.pkl'
    # with open(themePath + '\\' + themeEmbeddingFileName, 'rb') as f:
    #     themeEmbeddings3 = pickle.load(f)
    # themeEmbeddingFileName = 'theme_BERT_embeddings_400.pkl'
    # with open(themePath + '\\' + themeEmbeddingFileName, 'rb') as f:
    #     themeEmbeddings4 = pickle.load(f)
    # themeEmbeddingFileName = 'theme_BERT_embeddings_400p.pkl'
    # with open(themePath + '\\' + themeEmbeddingFileName, 'rb') as f:
    #     themeEmbeddings5 = pickle.load(f)
    # themeEmbeddings = themeEmbeddings1 + themeEmbeddings2 + themeEmbeddings3 + themeEmbeddings4 + themeEmbeddings5

    themeEmbeddingFileName = 'simplified_theme_BERT_embeddings.pkl'
    with open(themePath + '\\' + themeEmbeddingFileName, 'rb') as f:
        themeEmbeddings = pickle.load(f)

    lemmatizer = WordNetLemmatizer()
    nltk.download('wordnet')
    nltk.download('punkt')
    themeLemmaList = []
    for theme in themeList:
        # words = theme.split(' ')
        words = word_tokenize(theme)
        wordsLemma = ''
        for w in words:
            wordsLemma = wordsLemma + lemmatizer.lemmatize(w) + ' '
            # print(w, " : ", lemmatizer.lemmatize(w))
        themeLemmaList.append(wordsLemma[0:-1])
    
    sTokens = []
    for theme in themeLemmaList:
        sToken = [token for token in theme.split() if token in wv.vocab]
        sTokens = sTokens + sToken
    word_counts = map_word_frequency(sTokens)
    
    aveThemeEmbeddings = []
    for themeE in themeEmbeddings[0:100]:
        aveThemeE = get_weighted_feature_vectors(themeE,themeLemmaList)
        aveThemeEmbeddings.append(aveThemeE)
    
    # numTheme = len(clsEmbeddings)
    # similarityMat = np.ones((numTheme,numTheme))
    # for i in range(numTheme):
    #     for j in range(numTheme):
    #         if i == j:
    #             similarityMat[i,j] = -1
    #         else:
    #             similarityMat[i,j] = get_cosine_similarity(clsEmbeddings[i],clsEmbeddings[j])

    # print(similarityMat[0:30,0:30])

    # draw dendrogram from the theme embeddings
    linked = linkage(aveThemeEmbeddings[0:100], 'complete')
    labelList = themeList[0:100]

    plt.figure(figsize=(10, 7))
    dendrogram(linked,
                orientation='right',
                labels=labelList,
                distance_sort='descending',
                show_leaf_counts=True)
    plt.show()

    # with open('C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Name Entity Recognition\\themeEmbeddings.pkl', 'wb') as f:
    #     pickle.dump(themeEmbeds,f)
    # with open('C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Name Entity Recognition\\themeLemmaList.pkl', 'wb') as f:
    #     pickle.dump(themeLemmaList,f)

    # with open('C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Name Entity Recognition\\themeList.pkl', 'wb') as f:
    #     pickle.dump(themeList,f)

if __name__ == "__main__":    main()