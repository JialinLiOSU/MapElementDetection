# @author: Jialin Li
# @date: 10/26/2020
# @comments: generate choropleth map with vertical or horizontally aligned legend
# covering geographic regions of the world, US, Canada, Austrilia and Europe

# libaries ---------- basic
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from scipy.spatial import ConvexHull, Voronoi
import matplotlib.patches as mpatches



# libaries ---------- color scheme
# sequential
from matplotlib.cm import viridis
from matplotlib.cm import plasma
from matplotlib.cm import summer
from matplotlib.cm import gray
from matplotlib.cm import autumn
# diverging
from matplotlib.cm import PiYG
from matplotlib.cm import coolwarm
from matplotlib.cm import Spectral
from matplotlib.cm import BrBG
from matplotlib.cm import PRGn
# quantitative
from matplotlib.cm import Pastel1
from matplotlib.cm import Paired
from matplotlib.cm import Accent
from matplotlib.cm import Dark2
from matplotlib.cm import Set3

# NLTK libaraies, import corpus and extract frequent text
import nltk
from nltk.corpus import brown
nltk.download('brown')
nltk.download('universal_tagset')

# pandas, record the meta information

# define generation variables

# define generation variables

# configuration of map setting
isStateName = 0  # if show state name
isMainland = 1   # if show state that far from mainland: alaska
isRiver = 0      # if show
isCoastlines = 1  # if draw coastlines
isLat = 1        # if draw latitude
isLong = 1       # if draw longtitude
is3D = 0         # if draw 3D map

# configuration of map style
mapBackground = 0  # background color of the map
backgroundList = ['#FFFFFF', '#207ab0', '#333333',
                  '#aedba8', '#fffdc4', '#a8d5e3', '#d1d1d1']
mapMask = 0   # whether use map mask, stylish layer
serviceList = ["ESRI_Imagery_World_2D", 'Ocean_Basemap', "ESRI_StreetMap_World_2D",
               "NatGeo_World_Map", "NGS_Topo_US_2D",
               "NGS_Topo_US_2D", "World_Shaded_Relief", "World_Street_Map",
               "World_Topo_Map", "World_Terrain_Base", "canvas/World_Dark_Gray_Base",
               'Specialty/World_Navigation_Charts', 'Specialty/DeLorme_World_Base_Map',
               'Reference/World_Reference_Overlay',
               'Canvas/World_Light_Gray_Base', 'World_Physical_Map']  # map style
colorList = ['#FFFFFF', '#F7C353', '#CDCDCD', '#f3A581']
fontnameList = ['Courier New','Arial','Calibri','Times New Roman','Sans']

# configuration of visualization variables
mapPosition = 0      # center point of map
mapSize = 0          # bounding box size of the map
mapProjection = 0    # projection of the map (can be regarded as shape)
mapValue = 1.0         # opacity of the color
mapColor = 0     # color (hue) scheme of map
mapOrientation = 0   # transformation of map
mapTexture = 0       # textures on the map
texturePatterns = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]

# configuration of visualization elements
mapText = 0    # random selected text
showLegend = 0  # show map color legend

# meta_data = pd.read_csv('meta.csv', encoding='utf-8')

# us state name and acronym
short_state_names = {
    'AK': 'Alaska',
    'AL': 'Alabama',
    'AR': 'Arkansas',
    'AS': 'American Samoa',
    'AZ': 'Arizona',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
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
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
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
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

# extract 100 sentence from Brown corpus with 'government' topics
brown_sent = brown.sents(categories='government')[0:2000]
brown_title = []
for i in range(len(brown_sent)):
    title = ' '.join(brown_sent[i]).split(',')[0]
    if(len(title) < 80 and len(title) > 20):
        brown_title.append(title)
brown_title = brown_title[0:100]

# extract top 100 frequent words from Brown corpus
noundist = nltk.FreqDist(w2 for ((w1, t1), (w2, t2)) in
                         nltk.bigrams(brown.tagged_words(tagset="universal"))
                         if w1.lower() == "the" and t2 == "NOUN")
frequent_dist = noundist.most_common(100)
frequent_words = []
for i in range(len(frequent_dist)):
    frequent_words.append(frequent_dist[i][0])

def getFontName():
    a = random.randint(0,4)
    return fontnameList[a]

# parameters setting
def get_admin_level():
    a = random.uniform(0, 1)
    if (a <= 0.02):
        return 0
    else:
        return 1

# if show state name


def get_IsStateName():
    a = random.randint(1, 3)
    if(a < 2):
        return 0
    elif(a == 2):  # state name
        return 1
    elif(a == 3):  # randomly generate text
        return 2

# if show far land like hawaii and alaska (only for us)


def get_IsMainland():
    a = random.uniform(0, 1)
    if (a <= 0.5):
        return 0
    else:
        return 1

    # if show lat or long

# if show lat and long


def getLatLong():
    a = random.uniform(0, 1)
    if (a <= 0.7):
        return 0, 0
    else:
        return 1, 1

    # get the background color of the map


def getBackgroundColor():
    a = random.randint(1, 17)
    if (a <= 6):
        return backgroundList[0]
    elif (6 < a <= 12):
        return backgroundList[1]
    else:
        b = a - 12
        return backgroundList[b]

# get the style of map


def getStyle():
    a = random.randint(0, 15)
    return serviceList[a]

# get inherent style


def getInnerStyle():
    a = random.randint(1, 3)
    return a

# get the size


def getSize():
    a = random.randint(1, 8)
    if (a <= 2):
        return 0
    elif (2 < a <= 7):
        return 1
    else:
        return 2

# get the position


def getPosition(size):
    # small size
    if(size == 0):
        x1 = -168
        x2 = 185
        y1 = -random.uniform(65, 89)
        y2 = 89
        return x1, y1, x2, y2
    elif(size == 1):  # medium size
        x1 = -168
        x2 = 185
        y1 = -65
        y2 = 89
        return x1, y1, x2, y2
    else:  # large size
        x1 = -168
        x2 = 185
        y1 = -random.uniform(75, 89)
        y2 = 89
        return x1, y1, x2, y2

# get the position with style


def getStylePosition(size):
    # small size
    if(size == 0):
        x1 = -random.randint(136, 139)
        x2 = -random.randint(45, 50)
        y1 = random.randint(10, 15)
        y2 = random.randint(55, 60)
        return x1, y1, x2, y2
    elif(size == 1):  # medium size
        x1 = -random.randint(128, 131)
        x2 = -random.randint(60, 65)
        y1 = random.randint(18, 24)
        y2 = random.randint(50, 55)
        return x1, y1, x2, y2
    else:  # large size
        x1 = -random.randint(127, 129)
        x2 = -random.randint(62, 66)
        y1 = random.randint(20, 24)
        y2 = random.randint(50, 52)
        return x1, y1, x2, y2

# get the value


def getValue():
    opcacity = round(random.uniform(0.7, 1), 2)
    return opcacity

# get the value


def getStyleValue():
    opcacity = round(random.uniform(0.0, 0.1), 2)
    return opcacity

# get the color scheme
# 0: blank 1-3: single color, 4-8: sequential color
# 9-13: diverging color 14-18: quantitative color


def getcolor_scheme():
    a = random.randint(4, 18)
    print('color scheme: ' + str(a))
    return a

# get color, i: random number, a: color scheme


def getColor(i, a):
    if(a == 0):
        return colorList[0]
    elif(1 <= a <= 3):
        return colorList[a]
    elif(4 <= a <= 8):
        i = (i + random.randint(0, 2))*10
        if(a == 4):
            return viridis(i)
        elif(a == 5):
            return plasma(i)
        elif(a == 6):
            return summer(i)
        elif(a == 7):
            return gray(i)
        elif(a == 8):
            return autumn(i)
    elif(9 <= a <= 13):
        i = (i + random.randint(0, 2))*20
        if(a == 9):
            return PiYG(i)
        elif(a == 10):
            return coolwarm(i)
        elif(a == 11):
            return Spectral(i)
        elif(a == 12):
            return BrBG(i)
        elif(a == 13):
            return PRGn(i)
    elif(13 < a <= 18):
        i = i - random.randint(0, 4)
        if(i < 0):
            return '#FFFFFF'
        if(a == 14):
            return Pastel1(i)
        elif(a == 15):
            return Paired(i)
        elif(a == 16):
            return Accent(i)
        elif(a == 17):
            return Dark2(i)
        elif(a == 18):
            return Set3(i)

# if add texture


def isTexture():
    a = random.randint(1, 4)
    if (a <= 3):
        return 0
    else:
        return 1

# generate texture


def getTexture():
    b = random.randint(0, 9)  # texture pattern
    c = random.randint(1, 3)  # density of texture pattern
    d = random.randint(0, 1)  # if a state marked with texture
    if (d == 0):
        return ''
    else:
        return texturePatterns[b] * c

# generate title


def getTitle():
    a = random.randint(0, 1)
    title = ''
    if (a == 0):
        return title
    else:
        b = random.randint(0, 99)
        title = brown_title[b]
    if len(title) > 30:
        title = title[0:30]
    # title = '' # for Horizontal legend, don't add title
    return title

# generate text on state


def getText():
    a = random.randint(0, 99)
    return frequent_words[a]

# generate legend


def getLegend(a,numCol):
    labels = random.sample(range(0, 99), numCol)
    patchList = []
    for i in range(numCol):
        patch = mpatches.Patch(color=getColor(
        i * 2 + 1, a),  label=frequent_words[labels[i]])
        patchList.append(patch)
    return patchList

# get projection method
def getProjection():
    a = random.randint(0, 1)
    if(a == 0):
        return 'robin'
    else:
        return 'hammer'

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def setLegendSymbolShape(shapeCode):
    # no return value, just change the rect parameter of matplotlib
    if shapeCode == 0:
        l,h = 1,1
    elif shapeCode == 1:
        l,h = 2**0.5, 1
    elif shapeCode == 2:
        l,h = 2, 1
    else:
        l,h = 1,1
        print('Shape code is not correct')

    matplotlib.rcParams['legend.handlelength'] = l
    matplotlib.rcParams['legend.handleheight'] = h

path = 'C:\\Users\\jiali\\Desktop\\Map_Identification_Classification\\world map generation\\'
shpFileName = 'shpfile/cartogram/pop2007_0'
strList = []

# draw world map

def drawWmap(index, filename):

    # check aspect ratio
    asp_x = random.randint(7, 8)
    asp_y = random.randint(4, 5)

    fig = plt.figure(figsize=(8, 4), dpi=500)

    # 1. size and location
    mapSize = getSize()
    # x1, y1, x2, y2 = 73.62, 18.16, 134.76, 53.55 # china wgs84
    # x1, y1, x2, y2 = -124.70, 24.94, -66.97, 49.37 # US
    # x1, y1, x2, y2 = 126.11, 33.18, 129.58, 38.62 # South Korea
    # x1, y1, x2, y2 = -141.00, 41.67, -52.61, 83.11 # Canada
    # x1, y1, x2, y2 = 112.91, -43.66, 153.62, -10.71 # Austrilia
    x1, y1, x2, y2 = -10.47, 34.92, 40.17, 71.11 # Europe

    deltaX = x2 - x1
    deltaY = y2 - y1

    # map location and bounding box
    m = Basemap(lon_0=0, 
                projection='cyl', fix_aspect=True)
    # m.drawcoastlines(linewidth=0.25)
    # m.drawcountries(linewidth=0.25)

    # 2. administraitive level
    admin_level = 0

    ax = plt.gca()  # get current axes instance
    # ax = plt.Axes(fig, [0.25, 0.25, 0.75, 0.75], )

    # read polygon information from shape file, only show admin0 and admin1
    if (admin_level == 0):
        shp_info = m.readshapefile(
            path + shpFileName, 'state', drawbounds=True, linewidth=0.01)
        # 3. color scheme
        colorscheme = getcolor_scheme()
        # 4. if show text on each state
        isStateName = get_IsStateName()
        # 5. identify the text size
        font_size = random.randint(1, 2)
        # font_size = 0.5
        # 6. if add texture # no textures needed
        # mapTexture = isTexture()
        # 7. if draw Alaska and Hawaii
        isMainland = 1
        # 8. identify the opacity value
        opaVal = getValue()
        printed_names = []
        for info, shape in zip(m.state_info, m.state):
            # if (mapTexture == 1):
            #     poly = Polygon(shape, facecolor=getColor(len(info['CNTRY_NAME']), colorscheme),
            #                    edgecolor='k', alpha=opaVal, linewidth=0.5, hatch=getTexture())
            # else:
            poly = Polygon(shape, facecolor=getColor(len(info['CNTRY_NAME']), colorscheme),
                               alpha=opaVal, edgecolor='k', linewidth=0.05)

            ax.add_patch(poly)

            # add text on each state
            if (isStateName != 0):
                x, y = np.array(shape).mean(axis=0)
                hull = ConvexHull(shape)
                hull_points = np.array(shape)[hull.vertices]
                x, y = hull_points.mean(axis=0)
                short_name = info['CNTRY_NAME']
                if short_name in printed_names:
                    continue
                if (isStateName == 1):
                    plt.text(x + .1, y, short_name,
                             ha="center", fontsize=font_size)
                elif (isStateName == 2):
                    state_text = getText()
                    plt.text(x + .1, y, state_text,
                             ha="center", fontsize=font_size)
                printed_names += [short_name, ]

    # draw map

    # 9. if add long and lat
    isLat, isLong = getLatLong()
    # if (isLat == 1):
    #     margin = random.randint(2, 4) * 10
    #     m.drawparallels(np.arange(-90, 90, margin), labels=[1, 0, 0, 0], linewidth=0.2, fontsize=5)
    #     m.drawmeridians(np.arange(-180, 180, margin), labels=[0, 0, 0, 1], linewidth=0.2, fontsize=5)

    # 10. background color
    mapBackground = getBackgroundColor()
    ax.set_facecolor(mapBackground)

    # store the information into meta
    # plt.show()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # plt.show()
    plt.savefig(path+filename)
    # plt.savefig(path+filename,bbox_inches='tight',pad_inches=0.5)
    plt.close()
    original = Image.open(path+filename)
    width, height = original.size   # Get dimensions

    # left = (x1 - (-180)-deltaX/20 + 45)/360  * width  # us merc
    # top = (90 - y2 - deltaY/20 + 13) / 180 *height
    # right = (x2 - (-180)+deltaX/20 + 23)/360 * width
    # bottom = (90 - y1 + deltaY/20 + 8) / 180 * height 

    # left = (x1 - (-180)-deltaX/20 -25)/360  * width   # china merc
    # top = (90 - y2 - deltaY/20 +13) / 180 *height
    # right = (x2 - (-180)+deltaX/20 -49 )/360 * width
    # bottom = (90 - y1 + deltaY/20 +6 ) / 180 * height

    # left = (x1 - (-180)-deltaX/20 +10.5)/360  * width   # sk merc with china coordinates
    # top = (90 - y2 - deltaY/20 +28) / 180 *height
    # right = (x2 - (-180)+deltaX/20  -54)/360 * width
    # bottom = (90 - y1 + deltaY/20 -6 ) / 180 * height

    left = (x1 - (-180)-deltaX/20 )/360  * width   # standard cyl
    top = (90 - y2 - deltaY/20 ) / 180 *height
    right = (x2 - (-180)+deltaX/20 )/360 * width
    bottom = (90 - y1 + deltaY/20 ) / 180 * height
    # croppedImage = original.crop((left, top, right, bottom))

    croppedImage = original.crop((left, top, right, bottom))


    # rightImage.show()
    # leftImage.show()Image.show()
    # croppedImage.show()
    croppedImage.save(path+filename)

    # img = Image.open(path+filename)
    img = mpimg.imread(path+filename)
    w_img = img.shape[1]
    h_img = img.shape[0]
    # fig = plt.figure(dpi=150)
    asp_y = 5
    asp_x = asp_y/h_img*w_img
    fig = plt.figure(figsize=(asp_x, asp_y), dpi=150)
    r = fig.canvas.get_renderer()
    ax = plt.gca()  # get current axes instance
    
    # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
    #         hspace = 0, wspace = 0)
    # plt.margins(0,0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    imgplot = plt.imshow(img)

    # 11. if add title
    title = getTitle()
    titlePosition = random.randint(0, 1)
    # if titlePosition == 0:
    #     t = plt.title(title, y = 1.2)
    # else:
    #     t = plt.title(title, y=-0.2)
    # bb_t = t.get_window_extent(renderer=r)
    # width_t = bb_t.width
    # height_t = bb_t.height
    # print('width_t:',width_t)
    # print('height_t:',height_t)
    # plt.show()
    # 12. if add legends
    bbox_to_anchor_list=[(0.1, 1),(0.1, 0),(0.9,1),(0.9,0)] # position of legends
    numCol = random.randint(3, 6)
    legendSize = random.randint(6, 10)
    shapeCode = random.randint(0, 3)
    setLegendSymbolShape(shapeCode)

    if (colorscheme >= 4):
        showLegend = 1
        loc_var = random.randint(1, 4)
        if (loc_var == 1):
            patchList = getLegend(colorscheme, numCol)
            l = plt.legend(handles=patchList,bbox_to_anchor = bbox_to_anchor_list[0], loc='upper left',
                        prop={'size': legendSize},ncol = numCol, framealpha=1)
        elif (loc_var == 2):
            patchList = getLegend(colorscheme, numCol)
            l = plt.legend(handles=patchList,bbox_to_anchor = bbox_to_anchor_list[1],loc='lower left',
                        prop={'size': legendSize},ncol = numCol, framealpha=1)
        elif (loc_var == 3):
            patchList = getLegend(colorscheme, numCol)
            l = plt.legend(handles=patchList,bbox_to_anchor = bbox_to_anchor_list[2], loc='upper right',
                        prop={'size': legendSize},ncol = numCol, framealpha=1)
        elif (loc_var == 4):
            patchList = getLegend(colorscheme, numCol)
            l = plt.legend(handles=patchList,bbox_to_anchor = bbox_to_anchor_list[3],loc='lower right',
                        prop={'size': legendSize},ncol = numCol, framealpha=1)
    else:
        showLegend = 0
    plt.axis('off')
    # plt.show()

    

    # legend size
    if showLegend != 0:
        bb_l = l.get_window_extent(renderer=r)
        width_l = bb_l.width
        height_l = bb_l.height

    # figure size
    bb_fig = fig.get_window_extent(renderer = r)
    width_fig = bb_fig.width
    height_fig = bb_fig.height

    # add title position to strList
    # if title != '':
    #     x1 = bb_t.bounds[0] / width_fig
    #     y1 = bb_t.bounds[1] / height_fig
    #     wid = bb_t.bounds[2] / width_fig
    #     heig = bb_t.bounds[3] / height_fig
    #     strLine = filename + ',' + str(x1) + ',' + str(y1) + ',' + str(wid) + ',' + str(heig) + ',0' + '\n'
    #     strList.append(strLine)

    # add legend position to strList
    if showLegend != 0:
        x1 = bb_l.bounds[0] / width_fig
        y1 = bb_l.bounds[1] / height_fig
        wid = bb_l.bounds[2] / width_fig
        heig = bb_l.bounds[3] / height_fig
        strLine = filename + ',' + str(x1) + ',' + str(y1) + ',' + str(wid) + ',' + str(heig) + ',1' + '\n'
        strList.append(strLine)

    # remove borders
    
    
    plt.savefig(path+filename)
    # plt.savefig(path+filename,bbox_inches='tight',pad_inches = 0)
    plt.close()
    plt.show()

# draw world map with style
def drawMapUnclassed(index, filename):

    # check aspect ratio
    asp_x = random.randint(7, 8)
    asp_y = random.randint(4, 5)

    fig = plt.figure(figsize=(8, 4), dpi=500)

    # 1. size and location
    mapSize = getSize()
    # x1, y1, x2, y2 = 73.62, 18.16, 134.76, 53.55 # china wgs84
    x1, y1, x2, y2 = -124.70, 24.94, -66.97, 49.37 # US
    # x1, y1, x2, y2 = 126.11, 33.18, 129.58, 38.62 # South Korea
    # x1, y1, x2, y2 = -141.00, 41.67, -52.61, 83.11 # Canada
    # x1, y1, x2, y2 = 112.91, -43.66, 153.62, -10.71 # Austrilia
    # x1, y1, x2, y2 = -10.47, 34.92, 40.17, 71.11 # Europe

    deltaX = x2 - x1
    deltaY = y2 - y1

    # map location and bounding box
    m = Basemap(lon_0=0, 
                projection='cyl', fix_aspect=True)
    # m.drawcoastlines(linewidth=0.25)
    # m.drawcountries(linewidth=0.25)

    # 2. administraitive level
    admin_level = 0

    ax = plt.gca()  # get current axes instance
    # ax = plt.Axes(fig, [0.25, 0.25, 0.75, 0.75], )

    # read polygon information from shape file, only show admin0 and admin1
    if (admin_level == 0):
        shp_info = m.readshapefile(
            path + shpFileName, 'state', drawbounds=True, linewidth=0.01)
        # 3. color scheme
        colorscheme = getcolor_scheme()
        # 4. if show text on each state
        isStateName = get_IsStateName()
        # 5. identify the text size
        font_size = random.randint(1, 2)
        # font_size = 0.5
        # 6. if add texture # no textures needed
        # mapTexture = isTexture()
        # 7. if draw Alaska and Hawaii
        isMainland = 1
        # 8. identify the opacity value
        opaVal = getValue()
        printed_names = []
        for info, shape in zip(m.state_info, m.state):
            # if (mapTexture == 1):
            #     poly = Polygon(shape, facecolor=getColor(len(info['CNTRY_NAME']), colorscheme),
            #                    edgecolor='k', alpha=opaVal, linewidth=0.5, hatch=getTexture())
            # else:
            poly = Polygon(shape, facecolor=getColor(len(info['CNTRY_NAME']), colorscheme),
                               alpha=opaVal, edgecolor='k', linewidth=0.05)

            ax.add_patch(poly)

            # add text on each state
            if (isStateName != 0):
                x, y = np.array(shape).mean(axis=0)
                hull = ConvexHull(shape)
                hull_points = np.array(shape)[hull.vertices]
                x, y = hull_points.mean(axis=0)
                short_name = info['CNTRY_NAME']
                if short_name in printed_names:
                    continue
                if (isStateName == 1):
                    plt.text(x + .1, y, short_name,
                             ha="center", fontsize=font_size)
                elif (isStateName == 2):
                    state_text = getText()
                    plt.text(x + .1, y, state_text,
                             ha="center", fontsize=font_size)
                printed_names += [short_name, ]

    # draw map

    # 9. if add long and lat
    isLat, isLong = getLatLong()
    # if (isLat == 1):
    #     margin = random.randint(2, 4) * 10
    #     m.drawparallels(np.arange(-90, 90, margin), labels=[1, 0, 0, 0], linewidth=0.2, fontsize=5)
    #     m.drawmeridians(np.arange(-180, 180, margin), labels=[0, 0, 0, 1], linewidth=0.2, fontsize=5)

    # 10. background color
    mapBackground = getBackgroundColor()
    ax.set_facecolor(mapBackground)

    # store the information into meta
    # plt.show()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # plt.show()
    plt.savefig(path+filename)
    # plt.savefig(path+filename,bbox_inches='tight',pad_inches=0.5)
    plt.close()
    original = Image.open(path+filename)
    width, height = original.size   # Get dimensions

    # left = (x1 - (-180)-deltaX/20 + 45)/360  * width  # us merc
    # top = (90 - y2 - deltaY/20 + 13) / 180 *height
    # right = (x2 - (-180)+deltaX/20 + 23)/360 * width
    # bottom = (90 - y1 + deltaY/20 + 8) / 180 * height 

    # left = (x1 - (-180)-deltaX/20 -25)/360  * width   # china merc
    # top = (90 - y2 - deltaY/20 +13) / 180 *height
    # right = (x2 - (-180)+deltaX/20 -49 )/360 * width
    # bottom = (90 - y1 + deltaY/20 +6 ) / 180 * height

    # left = (x1 - (-180)-deltaX/20 +10.5)/360  * width   # sk merc with china coordinates
    # top = (90 - y2 - deltaY/20 +28) / 180 *height
    # right = (x2 - (-180)+deltaX/20  -54)/360 * width
    # bottom = (90 - y1 + deltaY/20 -6 ) / 180 * height

    left = (x1 - (-180)-deltaX/20 )/360  * width   # standard cyl
    top = (90 - y2 - deltaY/20 ) / 180 *height
    right = (x2 - (-180)+deltaX/20 )/360 * width
    bottom = (90 - y1 + deltaY/20 ) / 180 * height
    # croppedImage = original.crop((left, top, right, bottom))

    croppedImage = original.crop((left, top, right, bottom))


    # rightImage.show()
    # leftImage.show()Image.show()
    # croppedImage.show()
    croppedImage.save(path+filename)

    img = mpimg.imread(path+filename)
    fig = plt.figure(dpi=150)
    r = fig.canvas.get_renderer()
    ax = plt.gca()  # get current axes instance
    # fig = plt.figure(figsize=(asp_x, asp_y), dpi=150)
    imgplot = plt.imshow(img)

    # 11. if add title
    title = getTitle()
    titlePosition = random.randint(0, 1)
    if titlePosition == 0:
        t = plt.title(title, y = 1.1)
    else:
        t = plt.title(title, y=-0.1)
    bb_t = t.get_window_extent(renderer=r)
    width_t = bb_t.width
    height_t = bb_t.height
    # print('width_t:',width_t)
    # print('height_t:',height_t)
    # plt.show()
    # 12. if add legends
    bbox_to_anchor_list=[(0.1, 1),(0.1, 0),(0.9,1),(0.9,0)] # position of legends
    numCol = random.randint(3, 6)
    legendSize = random.randint(6, 10)
    shapeCode = random.randint(0, 3)
    setLegendSymbolShape(shapeCode)

    if (colorscheme >= 4):
        showLegend = 1
        loc_var = random.randint(1, 4)
        if (loc_var == 1):
            patchList = getLegend(colorscheme, numCol)
            l = plt.legend(handles=patchList,bbox_to_anchor = bbox_to_anchor_list[0], loc='lower left',
                        prop={'size': legendSize},ncol = numCol)
        elif (loc_var == 2):
            patchList = getLegend(colorscheme, numCol)
            l = plt.legend(handles=patchList,bbox_to_anchor = bbox_to_anchor_list[1],loc='upper left',
                        prop={'size': legendSize},ncol = numCol)
        elif (loc_var == 3):
            patchList = getLegend(colorscheme, numCol)
            l = plt.legend(handles=patchList,bbox_to_anchor = bbox_to_anchor_list[2], loc='lower right',
                        prop={'size': legendSize},ncol = numCol)
        elif (loc_var == 4):
            patchList = getLegend(colorscheme, numCol)
            l = plt.legend(handles=patchList,bbox_to_anchor = bbox_to_anchor_list[3],loc='upper right',
                        prop={'size': legendSize},ncol = numCol)
    else:
        showLegend = 0
    # plt.show()

    # legend size
    if showLegend != 0:
        bb_l = l.get_window_extent(renderer=r)
        width_l = bb_l.width
        height_l = bb_l.height

    # figure size
    bb_fig = fig.get_window_extent(renderer = r)
    width_fig = bb_fig.width
    height_fig = bb_fig.height

    # add title position to strList
    if title != '':
        x1 = bb_t.bounds[0] / width_fig
        y1 = bb_t.bounds[1] / height_fig
        wid = bb_t.bounds[2] / width_fig
        heig = bb_t.bounds[3] / height_fig
        strLine = filename + ',' + str(x1) + ',' + str(y1) + ',' + str(wid) + ',' + str(heig) + ',0' + '\n'
        strList.append(strLine)

    # add legend position to strList
    if showLegend != 0:
        x1 = bb_l.bounds[0] / width_fig
        y1 = bb_l.bounds[1] / height_fig
        wid = bb_l.bounds[2] / width_fig
        heig = bb_l.bounds[3] / height_fig
        strLine = filename + ',' + str(x1) + ',' + str(y1) + ',' + str(wid) + ',' + str(heig) + ',1' + '\n'
        strList.append(strLine)

    # remove borders
    plt.axis('off')
    plt.savefig(path+filename)
    plt.close()
    plt.show()

# generate map image


def main():
    
    for i in range(0,50):
        # for i in range(len(meta_data)):
        filename = 'generated_legend_EU_' + str(i) + '.png'
        # if(i >= 40 and i < 50):
        drawWmap(i, filename)
        # elif(i >= 15 and i < 30):
        #     drawWmapStyle(i,filename)
        # elif(i >= 25 and i < 50):
        #     # shpFileName = 'shpfile/cartogram/pop2007_0_us'
        #     drawWmap(i,filename)
        # elif(i >= 45 and i < 60):
        # drawWmapProjectionStyle(i,filename)

    # meta_data.to_csv('result.csv', index=False)
    filename='generated_Hlegend_annotation_EU'+'.txt'
    file = open(path + filename,'a')
    file.writelines(strList)
    # file.writelines(incorrectImgNameStrList)
    file.close() 


if __name__ == "__main__":
    main()
