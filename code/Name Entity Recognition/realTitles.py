# test finetuned NER model on real titles
realTitles = ['Share of high school students attending a school with a sworn law enforcement officer',
'Males per 100 Females, Census 2000','US population density', 'Average Temperature for the US States from July 2015',
'Population per square mile by state. 2000 census figures.','Average annual rainfall across the states of the United States of America',
'Estimated Median Household Income, 2008 Contiguous United States','Percent of People Below Poverty Level 2004',
'Obesity trends * Among U.S. Adults, BRFSS, 2010','1990 Census Data, % of Population 65 and Older',
'Choropleth map','Choropleth map - CIC returnable loans/borrows by US county',
'Thematic maps - choropleth maps', 'Change in Divorce Rates, Between 1980 and 1990',
'Figure 1., Percentage of the People Living in Poverty Areas by States: 2006-2010',
'state rankings','Total withdrawals and deliveries','Trump vote','Winning margins','Trade in goods with China as a % of state GDP',
'U.S. Motor Vehicle Fatalities, 2008','1170 Coronavirus (COVID-19) Cases in the US','Hazardous Waste Site Installations (1997)',
'Influenza Research Database Reported Cases 2017-18','COVID-19 in the U.S.','Mexican American Population, 2010 US Census',
'median income','states with most work stress','Unemployment 2008','Poverty in the United States','Cities supporting emissions reductions (455)',
'Q3 2018 Installed wind power capacity (MW)','Poverty in the United States', '2017 Poverty rate in the United States', 
'2011 US agriculture exports by state (Hover for breakdown)','Native American Alone/One or More Other Race','States Where Tim Has Spent Time',
'Minority group with highest percent of state population','Crime Rates in the US - 2003 vs. Election Results -2004',
'The Wild West Violent Crimes in the Western United States', 'Median Household Income in the United States: 2015',
'NBA players origins per capita', '48 states by population', 'Word happiness score (the higher the number, the happier)',
'Number of Persons per Wal-Mart Store', 'Geo Choropleth Chart: US Venture Capital Landscape 2001',
'U.S. Department of Agriculture - Honey Production, 2009-2013','Open Source Choropleth Maps',
'Federal Government Expenditure, Per Capita Ranges by State: Fiscal Year 2009', 'Kickstarter USA','Percent of Popl 65 and Older',
'U.S. Farmland','Sales by State', 'Well-Being index', 'Difference Map for SEA: RWM 1980 - RWM 1960',
'Food Insecurity Rate', 'Obesity Rate','HSU Alumni Per 100,000 People',
'Figure 2. Percentage of People in Poverty for the United States and Puerto Rico:2013','Regional Heat Map',
'The Number of Multi-racial Housholds per county in the Continental United States', 
'Percent of 4-year-olds Served by State Pre-K', 'One-year forecast change in jobs', 
'Percentage of People 25 Years and Over Who Have a Bachelor\'s Degree','Percentage of 18- to 24-year-olds overweight or obese',
'Choropleth, 5 Classes, Standard Deviation','Estimated % of adults who think global warming is happening, 2014',
'Death Rate from Drug Poisoning / Overdose', 'Rate of Temperature Change in the United States, 1901-2015']


trainTitles = ['District Wise Crime against women in India in 2015',]
import random
import numpy as np
import os
from spacy.util import minibatch, compounding
import spacy
from pathlib import Path
import plac
import en_core_web_sm

LABEL = "THEME"

import pickle
# path = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\code\\Name Entity Recognition\\labledTitles.pkl'
# TRAIN_DATA = pickle.load( open( path, "rb" ) )
nlpTest = en_core_web_sm.load()

# @plac.annotations(
#     model=(nlpTest, "option", "m", str),
#     new_model_name=("New model name for model meta.", "option", "nm", str),
#     output_dir=("Optional output directory", "option", "o", Path),
#     n_iter=("Number of training iterations", "option", "n", int),
# )
output_dir = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Name Entity Recognition\\trained model'
# nlp = spacy.load(model)  # load existing spaCy model
# # test the saved model
print("Loading from", output_dir)
nlp2 = spacy.load(output_dir)
        # Check the classes have loaded back consistently
        # assert nlp2.get_pipe("ner").move_names == move_names
        # doc2 = nlp2(test_text)
        # for ent in doc2.ents:
        #     print(ent.label_, ent.text)

for test_text in realTitles:
        doc = nlp2(test_text)
        print("Entities in '%s'" % test_text)
        for ent in doc.ents:
            print(ent.label_, ent.text)