# from the theme embeddings to calculate cos similarity
import pickle
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

def get_cosine_similarity(feature_vec_1, feature_vec_2):    
    return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]


def main():
    # load theme embeddings from pickle file
    with open('C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Name Entity Recognition\\themeEmbeddings.pkl', 'rb') as f:
        themeEmbeds = pickle.load(f)
    with open('C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Name Entity Recognition\\themeLemmaList.pkl', 'rb') as f:
        themeLemmaList = pickle.load(f)
    themeEmbedsProc = []
    themeLemmasProc = []
    numTheme = len(themeEmbeds)
    numthemeLemma = len(themeLemmaList)
    for i in range(numTheme):

        # remove the NaN records
        if math.isnan(themeEmbeds[i][0]) or len(themeEmbeds[i]) == 0:
            continue
        themeEmbedsProc.append(themeEmbeds[i])
        themeLemmasProc.append(themeLemmaList[i])
    
    numThemeProc = len(themeEmbedsProc)
    numthemeLemmaProc = len(themeLemmasProc)

    # similarityMat = np.ones((numTheme,numTheme))
    # for i in range(numThemeProc):
    #     for j in range(numThemeProc):
    #         if i == j:
    #             similarityMat[i,j] = -1
    #         else:
    #             similarityMat[i,j] = get_cosine_similarity(themeEmbedsProc[i],themeEmbedsProc[j])

    # print(similarityMat[0:20,0:20])

    linked = linkage(themeEmbedsProc[0:100], 'single')
    labelList = themeLemmasProc[0:100]

    plt.figure(figsize=(10, 7))
    dendrogram(linked,
                orientation='right',
                labels=labelList,
                distance_sort='descending',
                show_leaf_counts=True)
    plt.show()

    

    
    print('test')

   

if __name__ == "__main__":    main()