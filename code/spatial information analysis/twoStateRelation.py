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
import math

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

def calculateCent(stateGeometry):
    if stateGeometry['type'] == 'Polygon':
        coordinates = stateGeometry['coordinates'][0]
    elif stateGeometry['type'] == 'MultiPolygon':
        coordinates = stateGeometry['coordinates'][0][0]
    A = 0
    xmean = 0
    ymean = 0
    for i in range(len(coordinates) - 1):
        p1 = coordinates[i]
        p2 = coordinates[i+1]
        ai = p1[0] * p2[1] - p2[0] * p1[1]
        A += ai
        xmean += (p2[0] + p1[0]) * ai
        ymean += (p2[1] + p1[1]) * ai
    A = A / 2.0
    x = xmean / (6 * A)
    y = ymean / (6 * A)

    return [x,y]

def calculateOrient(point1, point2):
    radian = math.atan2(point2[1] - point1[1], point2[0] - point1[0])
    degree = math.degrees(radian)
    if (degree > 45 and degree <= 135) or (degree > -135 and degree <= -45):
        if point1[1] < point2[1]:
            return 'North'
        else:
            return 'South'
    elif (degree > -45 and degree <= 45) or (degree > 135 and degree <= 180) or (degree <= -135 and degree >= -180):
        if point1[0] < point2[0]:
            return 'East'
        else:
            return 'West'

if __name__ == "__main__":
    # two states
    state1 = 'Ohio'
    state2 = 'Indiana'
    # load state adjacency array to be used for adjacency relation identification
    adjFile = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\spatial information analysis\\USStateAdj.npy'
    stateAdjMat = np.load(adjFile)

    with open('C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\spatial information analysis\\stateNames.pickle', 'rb') as f:
        stateNames = pickle.load(f)
    # identify adjacency relation
    isadj = isAdj(state1, state2, stateAdjMat, stateNames)
    print(isadj)
    
    # identify orientation relation
    shpfname = r'C:\Users\jiali\Desktop\MapElementDetection\code\shpFiles\USA_ADM1.shp\USA_ADM1.shp'
    shp = shapex(shpfname)
    for f in shp:
        if f['properties']['NAME'] == state1:
            stateGeometry1 = f['geometry']
        if f['properties']['NAME'] == state2:
            stateGeometry2 = f['geometry']
    centroid1 = calculateCent(stateGeometry1)
    centroid2 = calculateCent(stateGeometry2)
    orient = calculateOrient(centroid1, centroid2)
    print(orient)

