"""
Identify the relationship between two states
The relations include adjacency, distance and orientation relation
"""

from osgeo import ogr
import numpy as np
import pickle
import sys
sys.path.append('C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\spatial information analysis')
from shapex import *

def isAdj(state1, state2, stateAdjMat, stateNames):
    idxState1 = findIndex(state1,stateNames)
    idxState2 = findIndex(state2,stateNames)
    if stateAdjMat[idxState1][idxState2] == 1:
        return True
    else:
        return False

def findIndex(stateName, stateNames):
    if not (stateName in stateNames):
        print("state name not found")
        return
    return stateNames.index(stateName)

if __name__ == "__main__":
    # two states
    state1 = 'Ohio'
    state2 = 'Texas'
    # load state adjacency array to be used for adjacency relation identification
    adjFile = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\spatial information analysis\\USStateAdj.npy'
    stateAdjMat = np.load(adjFile)

    with open('C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\spatial information analysis\\stateNames.pickle', 'rb') as f:
        stateNames = pickle.load(f)
    # identify adjacency relation
    isadj = isAdj(state1, state2, stateAdjMat, stateNames)

    print(isadj)
    