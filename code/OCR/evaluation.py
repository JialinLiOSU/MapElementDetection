results = ['Example charapletn man', 'lifeExp (2007)','States and UTs',
'TOPOGIAPHIC RELIEF','Mlales per Io0 Females','Choropleth Map',
'Measles incidence per district','1990 Census Data % of Population 65 and Older',
'Human Development Index (Statistics reported from UNDP 2O01)',
'Average rate per IO,000 people','Influenza Patents by Country According to USPTO data',
'Dasymetric Map of Annual Avergage DailyTraffic Density',
'Average Temperature for the US States from July 2015','Value of Land',
'Choropleth Map', 'cGISIn Demogra phlcs; Vlsuallzing Population Grouth western Mleration Gver zng-real Perlod',
'CIC Returnable LoanslBorrows byUS County','Percent ofhomes $3OO,OO0 and over',
'PHEDOMIMAMT MANUFACTUIING CTIVIT1, 1997','Percent of homes less than $50,000',
'New Zealand Suicide Rate by District Health Board (DHB)','World Life Expectancy Map',
'DENSITY OF POPULATI0I, 2011', 'POPULATIOYDENSITY, zang','Mycobacterium bovis in wildlife',
'Population % white ethnic groups 2001 Southampton output areas',
'DG ECHO Dally Map Emergency Response Coordination Centre (ERCC) COVID-19 pandemic worldwide',
'Population per square mile by state 2o00 census figures',
'Pcpulatian Density and Offerder Density Rates', 'Rhode Island Social Vulnerability',
'Median Household Income and Unrepaired Sinkholes', 'Life Expectancy Indian states 2011-2016, at birth',
'Population in Europe','LIBYA\'S POPULATION AND ENERGY PRODUCTION','Thematic maps choropleth maps',
'India Population Density Map',
'Minority group with highest percent of state population Excludes White nor Hispanic',
'MAP ES.1 The Estimated Eflects of Water Scarcityon GDP in Year 2O5O','Olnternational Mapping',
'Population in 2008','Map showing the HDI of countries',
'Difference in Proportion of Fire Area Inside Interface Minus Outside',
'2012 Population Estimates',
'Per Capita Income: Per capila income in he past 12 monlis fin 201 irilalion adjusted dallars|',
'Population density of Vancouver (by dissemination areas), 2011',
'Change in Divorce Rates Between 1980 and 1990',
'Crime Rates in the US 2003 vS. Election Results 2004','Number of Pedestrians',
'Religious Diversity in the U.S.,2010','2016 Median income in Pennsylvania counties'
]

truths = ['Example choropleth map','lifeExp (2007)','States and UTs',
'TOPOGIAPHIC RELIEF','Males per 100 Females','Choropleth Map',
'Measles incidence per district','1990 Census Data % of Population 65 and Older',
'Human Development Index (Statistics reported from UNDP 2001)',
'Average rate per 10,000 people','Influenza Patents by Country According to USPTO data',
'Dasymetric Map of Annual Avergage DailyTraffic Density',
'Average Temperature for the US States from July 2015','Value of Land',
'Choropleth Map', 'GIS In Demographics: Visualizing Population Growth & western Migration Over a 200-Year Period',
'CIC Returnable Loans/Borrows by US County','Percent of homes $300,000 and over',
'PREDOMIMANT MANUFACTURING ACTIVITY, 1997','Percent of homes less than $50,000',
'New Zealand Suicide Rate by District Health Board (DHB)','World Life Expectancy Map',
'DENSITY OF POPULATION, 2011','POPULATION DENSITY, 2000', 'Mycobacterium bovis in wildlife',
'Population % white ethnic groups 2001 Southampton output areas',
'DG ECHO Daily Map Emergency Response Coordination Centre (ERCC) COVID-19 pandemic worldwide',
'Population per square mile by state 2000 census figures',
'Populatian Density and Offender Density Rates','Rhode Island Social Vulnerability',
'Median Household Income and Unrepaired Sinkholes','Life Expectancy Indian states 2011-2016, at birth',
'Population in Europe','LIBYA\'S POPULATION AND ENERGY PRODUCTION','Thematic maps choropleth maps',
'India Population Density Map',
'Minority group with highest percent of state population Excludes White nor Hispanic',
'MAP ES.1 The Estimated Eflects of Water Scarcityon GDP in Year 2O5O','International Mapping',
'Population in 2008','Map showing the HDI of countries',
'Difference in Proportion of Fire Area Inside Interface Minus Outside',
'2012 Population Estimates',
'Per Capita Income: Per capita income in the past 12 months (in 2011 inflalion adjusted dallars)',
'Population density of Vancouver (by dissemination areas), 2011',
'Change in Divorce Rates Between 1980 and 1990',
'Crime Rates in the US 2003 vs. Election Results 2004','Number of Pedestrians',
'Religious Diversity in the U.S.,2010','2016 Median income in Pennsylvania counties'
]
print(len(results))
print(len(truths))
numResults =  len(results)
# title error rate
errorCount = 0
for i in range(numResults):
    if results[i] != truths[i]:
        errorCount = errorCount + 1
titleER = errorCount / numResults * 100
print('title error rate is: '+ str(titleER))

# mean word error rate
wordERList = []
for i in range(numResults):
    resultWordList = results[i].split(' ')
    truthWordList = truths[i].split(' ')
    numWord = len(truthWordList)
    errorWordCount = 0
    for truthWord in truthWordList:
        if truthWord not in resultWordList:
            errorWordCount = errorWordCount + 1
            continue
    wordER = errorWordCount / numWord * 100
    wordERList.append(wordER)
meanWordER = sum(wordERList) / len(wordERList)
print('mean word error rate is: '+ str(meanWordER))

# mean character error rate
charERList = []
for i in range(numResults):
    resultChars = results[i].replace(' ','')
    truthChars = truths[i].replace(' ','')
    numChar = len(truthChars)
    errorCharCount = 0
    for i in range(numChar):
        if i < len(resultChars):
            if resultChars[i] != truthChars[i]:
                errorCharCount = errorCharCount + 1
    charER = errorCharCount / numChar * 100
    charERList.append(charER)
meanCharER = sum(charERList) / len(charERList)
print('mean character error rate is: '+ str(meanCharER))
