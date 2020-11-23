# calculate text similarity
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

from gensim.models import Word2Vec
import numpy as np
# give a path of model to load function
from gensim.test.utils import common_texts
# word_emb_model = Word2Vec.load('word2vec.model')
# import gensim.downloader as api
# wv = api.load('word2vec-google-news-300')
print('test')
# word_emb_model = Word2Vec(sentences=common_texts, vector_size=300, window=5, min_count=1, workers=4)

# def map_word_frequency(document):
#     return Counter(itertools.chain(*document))
    
# def get_sif_feature_vectors(sentence1, sentence2, word_emb_model=wv):
#     sentence1 = [token for token in sentence1.split() if token in word_emb_model.vocab]
#     sentence2 = [token for token in sentence2.split() if token in word_emb_model.vocab]
#     word_counts = map_word_frequency((sentence1 + sentence2))
#     embedding_size = 300 # size of vectore in word embeddings
#     a = 0.001
#     sentence_set=[]
#     for sentence in [sentence1, sentence2]:
#         vs = np.zeros(embedding_size)
#         sentence_length = len(sentence)
#         for word in sentence:
#             a_value = a / (a + word_counts[word]) # smooth inverse frequency, SIF
#             vs = np.add(vs, np.multiply(a_value, word_emb_model[word])) # vs += sif * word_vector
#         vs = np.divide(vs, sentence_length) # weighted average
#         sentence_set.append(vs)
#     return sentence_set

# def get_word2vec_feature_vectors(sentence1, word_emb_model=wv):
#     sentence1 = [token for token in sentence1.split() if token in word_emb_model.vocab]
    

#     embedding_size = 300 # size of vectore in word embeddings

#     sentence_set=[]

#     vs = np.zeros(embedding_size)
#     sentence_length = len(sentence1)
#     for word in sentence1:
#         wordEmb = word_emb_model[word]
#         vs = np.add(vs, wordEmb) 
#     vs = np.divide(vs, sentence_length) # weighted average
#     return vs

def pre_process(corpus):
    # convert input corpus to lower case.
    corpus = corpus.lower()
    # collecting a list of stop words from nltk and punctuation form
    # string class and create single array.
    stopset = stopwords.words('english') + list(string.punctuation)
    # remove stop words and punctuations from string.
    # word_tokenize is used to tokenize the input corpus in word tokens.
    corpus = " ".join([i for i in word_tokenize(corpus) if i not in stopset])
    # remove non-ascii characters
    # corpus = unidecode(corpus)
    return corpus

def main():
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
            if '%' in theme:
                theme = theme.replace('%','percent ')
            if '$' in theme:
                theme = theme.replace('$',' ')
            if '(' in theme:
                theme = theme.replace('(',' ')
            if ')' in theme:
                theme = theme.replace(')',' ')
            if '  ' in theme:
                theme = theme.replace('  ',' ')
            theme = pre_process(theme)
            themeList.append(theme.lower())
    
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

    themeEmbeds = []
    # for tl in themeLemmaList:
    #     tlEmbed = get_word2vec_feature_vectors(tl)
    #     themeEmbeds.append(tlEmbed)
    print('test')

    # with open('C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Name Entity Recognition\\themeEmbeddings.pkl', 'wb') as f:
    #     pickle.dump(themeEmbeds,f)
    with open('C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Name Entity Recognition\\themeLemmaList.pkl', 'wb') as f:
        pickle.dump(themeLemmaList,f)

if __name__ == "__main__":    main()