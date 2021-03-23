# read the shapefiles of cartograms 
# to get the coordinate extent of each region

"""
Find out the states with the specified relation with one state
The relations include adjacency, distance and orientation relation
"""


import numpy as np
import pickle
import sys
sys.path.append(r'C:\Users\jiali\Desktop\Map_Identification_Classification\world map generation\getCartoCoordExtent')
from shapex import *
from geom.point import *
from geom.centroid import *

# us state name and acronym
short_state_names = {
    # 'AK': 'Alaska',
    'AL': 'Alabama',
    'AR': 'Arkansas',
    # 'AS': 'American Samoa',
    'AZ': 'Arizona',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        # 'GU': 'Guam',
        # 'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        # 'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',

        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        # 'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        # 'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

def getStateExtent(shp, country):
    for c in shp:
        x = c['properties']
        if c['properties']['NAME'] == country:
            break
    typeGeom = c['geometry']['type']
    coordGeom = c['geometry']['coordinates']
    minLat,maxLat, minLon, maxLon= 999999999, -999999999, 999999999, -999999999
    # if typeGeom != 'MultiPolygon':
    #     coordGeom = [coordGeom]
    
    for poly in coordGeom:
        if typeGeom != 'MultiPolygon':
            poly = [poly]
        tmpMinLon, tmpMaxLon = min(poly[0])[0], max(poly[0])[0]
        tmpMinLat, tmpMaxLat = min(poly[0], key = lambda t: t[1])[1], max(poly[0],key = lambda t: t[1])[1]
        if tmpMinLon < minLon:
            minLon = tmpMinLon
        if tmpMaxLon > maxLon:
            maxLon = tmpMaxLon
        if tmpMinLat < minLat:
            minLat = tmpMinLat
        if tmpMaxLat > maxLat:
            maxLat = tmpMaxLat

    return minLat,maxLat, minLon, maxLon

# get the point list of a state from the shapefile
def getPointList(shp, country):
    for c in shp:
        if c['properties']['NAME'] == country:
            print('test')
            break
    # c = [sfor s in shp]
    typeGeom = c['geometry']['type']
    coordGeom = c['geometry']['coordinates']

    if typeGeom != 'MultiPolygon':
#         print(coordGeom[0]) 
        coordList = coordGeom[0]
    else:
        lenList = [len(poly[0]) for poly in coordGeom]
#         print(lenList)
        index = lenList.index(max(lenList))
#         print(index)
#         print(coordGeom[index]) 
        coordList = coordGeom[index][0]
    return [ Point(p[0], p[1]) for p in coordList ]


if __name__ == "__main__":
    # path of the cartogram shapefiles
    # get geographic information from shape file
    shapefilePath = r'C:\Users\jiali\Desktop\MapElementDetection\code\shpFiles\USA_Contiguous_Albers_Equal_Area_Conic'

    fileName = 'USA_Contiguous_Albers_Equal_Area_Conic.shp'
    shp = shapex(shapefilePath + '\\' + fileName)

    stateNames = list(short_state_names.values())
    stateCentroidGeoCoordList = []
    for state in stateNames:
        # state = 'Ohio'
        pointList = getPointList(shp,state)
        centroidGeo = centroid(pointList)[1]

        xCentroidGeo = centroidGeo.x
        yCentroidGeo = centroidGeo.y
        stateCentroidGeoCoordList.append((state, (xCentroidGeo,yCentroidGeo)))
        
    
    with open(r'C:\Users\jiali\Desktop\MapElementDetection\code\state identification\stateCentroidGeoCoordsConic.pickle', 'wb') as f:
	    pickle.dump(USBoundResults,f)



