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

def map_word_frequency(document):
    return Counter(itertools.chain(*document))

# def get_sif_feature_vectors(sentence1, sentenceList, word_emb_model=wv):
#     sentence = [token for token in sentence1.split()]

#     sTokens = []
#     for s in sentenceList:
#         sToken = [token for token in s.split()]
#         sTokens = sTokens + sToken
#     # sentence2 = [token for token in sentence2.split() if token in word_emb_model.vocab]
#     word_counts = map_word_frequency(sTokens)
#     embedding_size = 300 # size of vectore in word embeddings
#     a = 0.001
   
#     vs = np.zeros(embedding_size)
#     sentence_length = len(sentence)
#     for word in sentence:
#         a_value = a / (a + word_counts[word]) # smooth inverse frequency, SIF
#         vs = np.add(vs, np.multiply(a_value, word_emb_model[word])) # vs += sif * word_vector
#     vs = np.divide(vs, sentence_length) # weighted average
#     return vs
    

# def get_word2vec_feature_vectors(sentence1, word_emb_model=wv):
#     sentence1 = [token for token in sentence1.split() ]
    
#     embedding_size = 300 # size of vectore in word embeddings

#     sentence_set=[]

#     vs = np.zeros(embedding_size)
#     sentence_length = len(sentence1)
#     for word in sentence1:
#         wordEmb = word_emb_model[word]
#         vs = np.add(vs, wordEmb) 
#     vs = np.divide(vs, sentence_length) # weighted average
#     return vs

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

    themePath = r'C:\Users\jiali\Desktop\MapElementDetection\code\Name Entity Recognition'
    # read theme list for labeling in dendrogram
    themeFileName = 'themeList.pkl'
    with open(themePath + '\\' + themeFileName, 'rb') as f:
        themeList = pickle.load(f)

    # read theme CLS embeddings from BERT model
    themeEmbeddingFileName = 'theme_BERT_embeddings_20.pkl'
    with open(themePath + '\\' + themeEmbeddingFileName, 'rb') as f:
        themeEmbeddings1 = pickle.load(f)

    themeEmbeddingFileName = 'theme_BERT_embeddings_30.pkl'
    with open(themePath + '\\' + themeEmbeddingFileName, 'rb') as f:
        themeEmbeddings2 = pickle.load(f)
    themeEmbeddings = themeEmbeddings1+themeEmbeddings2
    
    clsEmbeddings = []
    for themeE in themeEmbeddings:
        clsThemeE = themeE['[CLS]']
        clsEmbeddings.append(clsThemeE)
    
    numTheme = len(clsEmbeddings)
    similarityMat = np.ones((numTheme,numTheme))
    for i in range(numTheme):
        for j in range(numTheme):
            if i == j:
                similarityMat[i,j] = -1
            else:
                similarityMat[i,j] = get_cosine_similarity(clsEmbeddings[i],clsEmbeddings[j])

    print(similarityMat[0:30,0:30])

    # draw dendrogram from the theme embeddings
    linked = linkage(clsEmbeddings, 'complete')
    labelList = themeList[0:30]

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