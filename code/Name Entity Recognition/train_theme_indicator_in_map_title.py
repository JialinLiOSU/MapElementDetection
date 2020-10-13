from __future__ import unicode_literals, print_function

import random
import numpy as np
import os
from spacy.util import minibatch, compounding
import spacy
from pathlib import Path
import plac
import en_core_web_sm
from spacy.gold import GoldParse
from spacy.scorer import Scorer

# new entity label
theme = "THEME"
region = "GPE"
time = "DATE"
admin = "ADMIN"
# training data
# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting

import pickle
# path = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\code\\Name Entity Recognition\\labledTitles.pkl'
# TRAIN_DATA = pickle.load( open( path, "rb" ) )
nlpTest = en_core_web_sm.load()

def evaluate(ner_model, examples):
    scorer = Scorer()
    for input_, annot in examples:
        doc_gold_text = ner_model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annot['entities'])
        pred_value = ner_model(input_)
        scorer.score(pred_value, gold)
    return scorer.scores

TRAIN_DATA = [(
"Sales, receipts, or value of shipments of firms in 1959 by province ",
{"entities": [(0, 47, theme),(51, 55, time),(59, 67, admin)]},
),
(
"By province in the United States in 1982 Elementary-secondary revenue from transportation programs ",
{"entities": [(3, 11, admin),(15, 32, region),(36, 40, time),(41, 98, theme)]},
),
(
"Number of hospitals by province ",
{"entities": [(0, 19, theme),(23, 31, admin)]},
),
(
"In 2012 Average square footage of houses ",
{"entities": [(3, 7, time),(8, 40, theme)]},
),
(
"Average hours per day spent on Attending or hosting social events in US by census tract in 1995 ",
{"entities": [(0, 65, theme),(69, 71, region),(75, 87, admin),(91, 95, time)]},
),
(
"By county in 1976 Average hours per day spent on Travel related to organizational, civic, and religious activities ",
{"entities": [(3, 9, admin),(13, 17, time),(18, 114, theme)]},
),
(
"Federal government expenditure (per capita) by census tract ",
{"entities": [(0, 43, theme),(47, 59, admin)]},
),
(
"By state Elementary-secondary revenue from state sources ",
{"entities": [(3, 8, admin),(9, 56, theme)]},
),
(
"In UK percent of households above $200k in 1968 by province ",
{"entities": [(3, 5, region),(6, 39, theme),(43, 47, time),(51, 59, admin)]},
),
(
"In 2013 Average hours per day spent on Social service and care activities  in the United States by county ",
{"entities": [(3, 7, time),(8, 74, theme),(78, 95, region),(99, 105, admin)]},
),
(
"By county number of fire points ",
{"entities": [(3, 9, admin),(10, 31, theme)]},
),
(
"In 1975 Production workers annual hours of Ship and boat building in South Korea by state ",
{"entities": [(3, 7, time),(8, 65, theme),(69, 80, region),(84, 89, admin)]},
),
(
"In the United States Average square footage of houses by township ",
{"entities": [(3, 20, region),(21, 53, theme),(57, 65, admin)]},
),
(
"By state Estimated annual sales for Womens clothing stores in 2004 ",
{"entities": [(3, 8, admin),(9, 58, theme),(62, 66, time)]},
),
(
"Gross domestic income (nominal or ppp) per capita by census tract in the United States ",
{"entities": [(0, 49, theme),(53, 65, admin),(69, 86, region)]},
),
(
"Population density of people enrolled in College or graduate school in U.S. ",
{"entities": [(0, 67, theme),(71, 75, region)]},
),
(
"In U.S. Elementary-secondary revenue from school lunch charges ",
{"entities": [(3, 7, region),(8, 62, theme)]},
),
(
"Elementary-secondary revenue from transportation programs in 1998 ",
{"entities": [(0, 57, theme),(61, 65, time)]},
),
(
"Average percent of time engaged in by womenHousehold services in 1995 ",
{"entities": [(0, 61, theme),(65, 69, time)]},
),
(
"By county in UK in 1976 energy consumption (per capita) ",
{"entities": [(3, 9, admin),(13, 15, region),(19, 23, time),(24, 55, theme)]},
),
(
"In UK Estimated annual sales for total (excl. motor vehicle & parts & gasoline stations) ",
{"entities": [(3, 5, region),(6, 88, theme)]},
),
(
"In 1973 Percent change of All Race ",
{"entities": [(3, 7, time),(8, 34, theme)]},
),
(
"In UK Estimated annual sales for Nonstore retailers ",
{"entities": [(3, 5, region),(6, 51, theme)]},
),
(
"Average hours per day spent on Caring for household adults in USA in 1987 ",
{"entities": [(0, 58, theme),(62, 65, region),(69, 73, time)]},
),
(
"In France Number of firms in 1957 ",
{"entities": [(3, 9, region),(10, 25, theme),(29, 33, time)]},
),
(
"By province Sales, receipts, or value of shipments of firms in U.S. ",
{"entities": [(3, 11, admin),(12, 59, theme),(63, 67, region)]},
),
(
"In 2007 by county Households with one or more people 65 years and over in US ",
{"entities": [(3, 7, time),(11, 17, admin),(18, 70, theme),(74, 76, region)]},
),
(
"In 1969 Average percent of time engaged in by womenEating and drinking ",
{"entities": [(3, 7, time),(8, 70, theme)]},
),
(
"In 1993 adult obesity ",
{"entities": [(3, 7, time),(8, 21, theme)]},
),
(
"In 2005 Elementary-secondary revenue from special education in UK ",
{"entities": [(3, 7, time),(8, 59, theme),(63, 65, region)]},
),
(
"By township Average percent of time engaged in Playing games in UK ",
{"entities": [(3, 11, admin),(12, 60, theme),(64, 66, region)]},
),
(
"By county rate of male ",
{"entities": [(3, 9, admin),(10, 22, theme)]},
),
(
"By county in U.S. Exports value of firms in 2009 ",
{"entities": [(3, 9, admin),(13, 17, region),(18, 40, theme),(44, 48, time)]},
),
(
"Helicobacter pylori rate by township in China ",
{"entities": [(0, 24, theme),(28, 36, admin),(40, 45, region)]},
),
(
"In France in 1983 by township Average number of bedrooms of houses ",
{"entities": [(3, 9, region),(13, 17, time),(21, 29, admin),(30, 66, theme)]},
),
(
"In the United States number of schools by state ",
{"entities": [(3, 20, region),(21, 38, theme),(42, 47, admin)]},
),
(
"In 1957 Average household size by county ",
{"entities": [(3, 7, time),(8, 30, theme),(34, 40, admin)]},
),
(
"In 1953 Percent of population of widowed in South Korea by county ",
{"entities": [(3, 7, time),(8, 40, theme),(44, 55, region),(59, 65, admin)]},
),
(
"By state in 1994 License plate vanitization rate ",
{"entities": [(3, 8, admin),(12, 16, time),(17, 48, theme)]},
),
(
"In 2002 in the United States percent of houses with annual income of $300,000 and over by province ",
{"entities": [(3, 7, time),(11, 28, region),(29, 86, theme),(90, 98, admin)]},
),
(
"Difference in number of people of who believe climate change in U.S. in 2012 ",
{"entities": [(0, 60, theme),(64, 68, region),(72, 76, time)]},
),
(
"Percent change of people who are alumni of OSU by township ",
{"entities": [(0, 46, theme),(50, 58, admin)]},
),
(
"Number of people of people enrolled in College or graduate school by county in South Korea in 1979 ",
{"entities": [(0, 65, theme),(69, 75, admin),(79, 90, region),(94, 98, time)]},
),
(
"By state in 1961 Total capital expenditures of Chemical manufacturing ",
{"entities": [(3, 8, admin),(12, 16, time),(17, 69, theme)]},
),
(
"Utility expenditure of governments in 2006 ",
{"entities": [(0, 34, theme),(38, 42, time)]},
),
(
"In 2006 by province in France Interest on general debt of governments ",
{"entities": [(3, 7, time),(11, 19, admin),(23, 29, region),(30, 69, theme)]},
),
(
"By state in USA Other taxes of governments ",
{"entities": [(3, 8, admin),(12, 15, region),(16, 42, theme)]},
),
(
"By county Annual payroll in France in 2003 ",
{"entities": [(3, 9, admin),(10, 24, theme),(28, 34, region),(38, 42, time)]},
),
(
"By state difference in population density of frauds ",
{"entities": [(3, 8, admin),(9, 51, theme)]},
),
(
"In China Agriculture exports in 1993 by province ",
{"entities": [(3, 8, region),(9, 28, theme),(32, 36, time),(40, 48, admin)]},
),
(
"In China License Taxes of governments by county in 1968 ",
{"entities": [(3, 8, region),(9, 37, theme),(41, 47, admin),(51, 55, time)]},
),
(
"In 2002 Production workers annual wages of Other chemical product and preparation manufacturing by state ",
{"entities": [(3, 7, time),(8, 95, theme),(99, 104, admin)]},
),
(
"Estimated annual sales for Pharmacies & drug stores in 1981 by census tract ",
{"entities": [(0, 51, theme),(55, 59, time),(63, 75, admin)]},
),
(
"In 2008 crime rate ",
{"entities": [(3, 7, time),(8, 18, theme)]},
),
(
"In 1966 by county in U.S. Production workers average for year of Transportation equipment manufacturing ",
{"entities": [(3, 7, time),(11, 17, admin),(21, 25, region),(26, 103, theme)]},
),
(
"In the United States in 1964 Total value of shipments and receipts for services of Alumina and aluminum production and processing by census tract ",
{"entities": [(3, 20, region),(24, 28, time),(29, 129, theme),(133, 145, admin)]},
),
(
"By county Average number of bedrooms of houses ",
{"entities": [(3, 9, admin),(10, 46, theme)]},
),
(
"Difference in population density of people who are alumni of OSU in 1978 in China by province ",
{"entities": [(0, 64, theme),(68, 72, time),(76, 81, region),(85, 93, admin)]},
),
(
"License taxes of governments by county ",
{"entities": [(0, 28, theme),(32, 38, admin)]},
),
(
"In China by county in 1979 Family households (families) ",
{"entities": [(3, 8, region),(12, 18, admin),(22, 26, time),(27, 55, theme)]},
),
(
"Current spending of elementary-secondary expenditure by census tract in the United States in 1955 ",
{"entities": [(0, 52, theme),(56, 68, admin),(72, 89, region),(93, 97, time)]},
),
(
"By province Estimated annual sales for Elect. shopping & m/o houses ",
{"entities": [(3, 11, admin),(12, 67, theme)]},
),
(
"Direct expenditure of governments in 2019 in UK by census tract ",
{"entities": [(0, 33, theme),(37, 41, time),(45, 47, region),(51, 63, admin)]},
),
(
"In 1950 by province Average hours per day by men spent on Travel related to personal care] ",
{"entities": [(3, 7, time),(11, 19, admin),(20, 90, theme)]},
),
(
"In the United States by township Households with female householder, no husband present, family in 1951 ",
{"entities": [(3, 20, region),(24, 32, admin),(33, 95, theme),(99, 103, time)]},
),
(
"Total capital expenditures of Apparel knitting mills in Canada ",
{"entities": [(0, 52, theme),(56, 62, region)]},
),
(
"By county Gross profit of companies in UK ",
{"entities": [(3, 9, admin),(10, 35, theme),(39, 41, region)]},
),
(
"In 1976 Family households (families) ",
{"entities": [(3, 7, time),(8, 36, theme)]},
),
(
"By state in US Households with female householder, no husband present, family in 1959 ",
{"entities": [(3, 8, admin),(12, 14, region),(15, 77, theme),(81, 85, time)]},
),
(
"Unemployment rate in France by state ",
{"entities": [(0, 17, theme),(21, 27, region),(31, 36, admin)]},
),
(
"In China annual average precipitation ",
{"entities": [(3, 8, region),(9, 37, theme)]},
),
(
"Total capital expenditures of Paint, coating, and adhesive manufacturing in 2003 ",
{"entities": [(0, 72, theme),(76, 80, time)]},
),
(
"In 1955 Elementary-secondary revenue from compensatory programs in the United States ",
{"entities": [(3, 7, time),(8, 63, theme),(67, 84, region)]},
),
(
"Total value of shipments and receipts for services of Other chemical product and preparation manufacturing in 2004 in US ",
{"entities": [(0, 106, theme),(110, 114, time),(118, 120, region)]},
),
(
"In South Korea Estimated annual sales for Nonstore retailers by county in 2009 ",
{"entities": [(3, 14, region),(15, 60, theme),(64, 70, admin),(74, 78, time)]},
),
(
"In 2005 Total capital expenditures of Tobacco manufacturing by census tract ",
{"entities": [(3, 7, time),(8, 59, theme),(63, 75, admin)]},
),
(
"By township in 1996 in the United States Total value of shipments and receipts for services of Other wood product manufacturing ",
{"entities": [(3, 11, admin),(15, 19, time),(23, 40, region),(41, 127, theme)]},
),
(
"By province Average hours per day by women spent on Caring for and helping household members in France ",
{"entities": [(3, 11, admin),(12, 92, theme),(96, 102, region)]},
),
(
"In China in 1952 Number of employees of Soap, cleaning compound, and toilet preparation manufacturing by province ",
{"entities": [(3, 8, region),(12, 16, time),(17, 101, theme),(105, 113, admin)]},
),
(
"By census tract in UK Elementary-secondary revenue from local government in 1958 ",
{"entities": [(3, 15, admin),(19, 21, region),(22, 72, theme),(76, 80, time)]},
),
(
"In 1996 General sales of governments by census tract in US ",
{"entities": [(3, 7, time),(8, 36, theme),(40, 52, admin),(56, 58, region)]},
),
(
"By census tract in 1976 Estimated annual sales for Food services & drinking places in UK ",
{"entities": [(3, 15, admin),(19, 23, time),(24, 82, theme),(86, 88, region)]},
),
(
"In the United States in 1953 Average square footage of houses ",
{"entities": [(3, 20, region),(24, 28, time),(29, 61, theme)]},
),
(
"Number of paid employees in South Korea ",
{"entities": [(0, 24, theme),(28, 39, region)]},
),
(
"Percent change of people enrolled in Nursery school, people enrolled in preschool in 2006 by census tract ",
{"entities": [(0, 81, theme),(85, 89, time),(93, 105, admin)]},
),
(
"Utility revenue of governments by county ",
{"entities": [(0, 30, theme),(34, 40, admin)]},
),
(
"Households with one or more people under 18 years by census tract ",
{"entities": [(0, 49, theme),(53, 65, admin)]},
),
(
"In US Production workers annual wages of Transportation equipment manufacturing in 1953 by census tract ",
{"entities": [(3, 5, region),(6, 79, theme),(83, 87, time),(91, 103, admin)]},
),
(
"By state Production workers average for year of Forging and stamping,Cutlery and handtool manufacturing ",
{"entities": [(3, 8, admin),(9, 103, theme)]},
),
(
"In 1995 by state Number of paid employees ",
{"entities": [(3, 7, time),(11, 16, admin),(17, 41, theme)]},
),
(
"Households with male householder, no wife present, family in 2000 in U.S. ",
{"entities": [(0, 57, theme),(61, 65, time),(69, 73, region)]},
),
(
"Average price for honey per pound in USA ",
{"entities": [(0, 33, theme),(37, 40, region)]},
),
(
"By census tract Elementary-secondary revenue from federal sources in UK ",
{"entities": [(3, 15, admin),(16, 65, theme),(69, 71, region)]},
),
(
"In 2017 in UK Number of paid employees ",
{"entities": [(3, 7, time),(11, 13, region),(14, 38, theme)]},
),
(
"Population density of separated in 2014 in Canada ",
{"entities": [(0, 31, theme),(35, 39, time),(43, 49, region)]},
),
(
"By county CO2 emission (per capita) ",
{"entities": [(3, 9, admin),(10, 35, theme)]},
),
(
"Sales, receipts, or value of shipments of firms in 1979 by census tract ",
{"entities": [(0, 47, theme),(51, 55, time),(59, 71, admin)]},
),
(
"In 2015 Married-couple family ",
{"entities": [(3, 7, time),(8, 29, theme)]},
),
(
"In the United States Other taxes of governments in 1989 ",
{"entities": [(3, 20, region),(21, 47, theme),(51, 55, time)]},
),
(
"By township General revenue of governments in 2009 ",
{"entities": [(3, 11, admin),(12, 42, theme),(46, 50, time)]},
),
(
"In China by state Number of paid employees ",
{"entities": [(3, 8, region),(12, 17, admin),(18, 42, theme)]},
),
(
"In 1976 Average square footage of houses ",
{"entities": [(3, 7, time),(8, 40, theme)]},
),
(
"In 2001 by township Current spending of elementary-secondary expenditure ",
{"entities": [(3, 7, time),(11, 19, admin),(20, 72, theme)]},
),
(
"In 2020 in USA Elementary-secondary revenue from other state aid by state ",
{"entities": [(3, 7, time),(11, 14, region),(15, 64, theme),(68, 73, admin)]},
),
(
"By province Number of paid employees in 2011 in the United States ",
{"entities": [(3, 11, admin),(12, 36, theme),(40, 44, time),(48, 65, region)]},
),
(
"Freedom index in US in 1982 by county ",
{"entities": [(0, 13, theme),(17, 19, region),(23, 27, time),(31, 37, admin)]},
),
(
"In 1961 number of fire points in UK ",
{"entities": [(3, 7, time),(8, 29, theme),(33, 35, region)]},
),
(
"Elementary-secondary revenue from other state aid in U.S. ",
{"entities": [(0, 49, theme),(53, 57, region)]},
),
(
"By province Population density of people enrolled in Elementary school (grades 1-8) in the United States in 1977 ",
{"entities": [(3, 11, admin),(12, 83, theme),(87, 104, region),(108, 112, time)]},
),
(
"In the United States by province in 2008 Elementary-secondary revenue from federal sources ",
{"entities": [(3, 20, region),(24, 32, admin),(36, 40, time),(41, 90, theme)]},
),
(
"In 1976 Average hours per day by men spent on Indoor and outdoor maintenance, building, and cleanup activities by county ",
{"entities": [(3, 7, time),(8, 110, theme),(114, 120, admin)]},
),
(
"In 1978 by township Average monthly housing cost ",
{"entities": [(3, 7, time),(11, 19, admin),(20, 48, theme)]},
),
(
"In South Korea by census tract Average number of bedrooms of houses in 2015 ",
{"entities": [(3, 14, region),(18, 30, admin),(31, 67, theme),(71, 75, time)]},
),
(
"In 1950 Estimated annual sales for Furniture & home furn. Stores by township in France ",
{"entities": [(3, 7, time),(8, 64, theme),(68, 76, admin),(80, 86, region)]},
),
(
"By township Current spending of elementary-secondary expenditure ",
{"entities": [(3, 11, admin),(12, 64, theme)]},
),
(
"Difference in population density of people enrolled in Nursery school, people enrolled in preschool in USA ",
{"entities": [(0, 99, theme),(103, 106, region)]},
),
(
"Number of earthquake by census tract ",
{"entities": [(0, 20, theme),(24, 36, admin)]},
),
(
"In 1978 Households with householder living alone ",
{"entities": [(3, 7, time),(8, 48, theme)]},
),
(
"By census tract freedom index ",
{"entities": [(3, 15, admin),(16, 29, theme)]},
),
(
"Price of land in UK in 2015 ",
{"entities": [(0, 13, theme),(17, 19, region),(23, 27, time)]},
),
(
"By state in US Average percent of time engaged in by menSports, exercise, and recreation in 1996 ",
{"entities": [(3, 8, admin),(12, 14, region),(15, 88, theme),(92, 96, time)]},
),
(
"Average year built in the United States by census tract ",
{"entities": [(0, 18, theme),(22, 39, region),(43, 55, admin)]},
),
(
"Federal government expenditure (per capita) in USA ",
{"entities": [(0, 43, theme),(47, 50, region)]},
),
(
"In US Estimated annual sales for New car dealers ",
{"entities": [(3, 5, region),(6, 48, theme)]},
),
(
"Estimated annual sales for Home furnishings stores by census tract in U.S. ",
{"entities": [(0, 50, theme),(54, 66, admin),(70, 74, region)]},
),
(
"In 1968 by province in Canada economic growth rate ",
{"entities": [(3, 7, time),(11, 19, admin),(23, 29, region),(30, 50, theme)]},
),
(
"Average household size in 1976 by census tract in Canada ",
{"entities": [(0, 22, theme),(26, 30, time),(34, 46, admin),(50, 56, region)]},
),
(
"Percent of population of People whose native language is Russian by census tract in China in 2009 ",
{"entities": [(0, 64, theme),(68, 80, admin),(84, 89, region),(93, 97, time)]},
),
(
"In 1989 import and export statistics ",
{"entities": [(3, 7, time),(8, 36, theme)]},
),
(
"Estimated annual sales for Home furnishings stores in UK in 1953 ",
{"entities": [(0, 50, theme),(54, 56, region),(60, 64, time)]},
),
(
"Number of fire points by township in 1984 in China ",
{"entities": [(0, 21, theme),(25, 33, admin),(37, 41, time),(45, 50, region)]},
),
(
"GDP (nominal or ppp) in USA ",
{"entities": [(0, 20, theme),(24, 27, region)]},
),
(
"In China in 1967 Elementary-secondary revenue from special education ",
{"entities": [(3, 8, region),(12, 16, time),(17, 68, theme)]},
),
(
"Nonfamily households in 2000 by county ",
{"entities": [(0, 20, theme),(24, 28, time),(32, 38, admin)]},
),
(
"By state number of fire points in 1989 ",
{"entities": [(3, 8, admin),(9, 30, theme),(34, 38, time)]},
),
(
"Average hours per day by women spent on Taking class for degree, certificate, or licensure in 2015 ",
{"entities": [(0, 90, theme),(94, 98, time)]},
),
(
"In UK in 1986 Estimated annual sales for Pharmacies & drug stores by township ",
{"entities": [(3, 5, region),(9, 13, time),(14, 65, theme),(69, 77, admin)]},
),
(
"Elementary-secondary revenue by state in 1966 ",
{"entities": [(0, 28, theme),(32, 37, admin),(41, 45, time)]},
),
(
"In US by township Estimated annual sales for Department stores in 2003 ",
{"entities": [(3, 5, region),(9, 17, admin),(18, 62, theme),(66, 70, time)]},
),
(
"In 1974 Annual payroll ",
{"entities": [(3, 7, time),(8, 22, theme)]},
),
(
"In USA number of fire points by province ",
{"entities": [(3, 6, region),(7, 28, theme),(32, 40, admin)]},
),
(
"By county in 1999 Estimated annual sales for Sporting goods hobby, musical instrument, & book stores in USA ",
{"entities": [(3, 9, admin),(13, 17, time),(18, 100, theme),(104, 107, region)]},
),
(
"In 1968 Estimated annual sales for Electronics & appliance stores in Canada by province ",
{"entities": [(3, 7, time),(8, 65, theme),(69, 75, region),(79, 87, admin)]},
),
(
"Taxes of governments in 1989 by township ",
{"entities": [(0, 20, theme),(24, 28, time),(32, 40, admin)]},
),
(
"Average monthly housing cost as percentage of income in 1987 in U.S. ",
{"entities": [(0, 52, theme),(56, 60, time),(64, 68, region)]},
),
(
"In 2012 Household income by county in Canada ",
{"entities": [(3, 7, time),(8, 24, theme),(28, 34, admin),(38, 44, region)]},
),
(
"Number of employees of Motor vehicle manufacturing by township in Canada ",
{"entities": [(0, 50, theme),(54, 62, admin),(66, 72, region)]},
),
(
"In 2009 Average hours per day spent on Religious and spiritual activities ",
{"entities": [(3, 7, time),(8, 73, theme)]},
),
(
"Exports value of firms in China ",
{"entities": [(0, 22, theme),(26, 31, region)]},
),
(
"Elementary-secondary revenue from special education in 1951 in the United States by province ",
{"entities": [(0, 51, theme),(55, 59, time),(63, 80, region),(84, 92, admin)]},
),
(
"In 1957 difference in number of people of White by census tract in U.S. ",
{"entities": [(3, 7, time),(8, 47, theme),(51, 63, admin),(67, 71, region)]},
),
(
"In 2016 by township in the United States Family households with own children of the householder under 18 years ",
{"entities": [(3, 7, time),(11, 19, admin),(23, 40, region),(41, 110, theme)]},
),
(
"Average percent of time engaged in by menAppliances, tools, and toys in 1995 ",
{"entities": [(0, 68, theme),(72, 76, time)]},
),
(
"In Canada Production workers annual hours of Other fabricated metal product manufacturing by province ",
{"entities": [(3, 9, region),(10, 89, theme),(93, 101, admin)]},
),
(
"In US by province number of patent ",
{"entities": [(3, 5, region),(9, 17, admin),(18, 34, theme)]},
),
(
"In 1978 energy consumption (per capita) in UK ",
{"entities": [(3, 7, time),(8, 39, theme),(43, 45, region)]},
),
(
"By census tract homicide rate ",
{"entities": [(3, 15, admin),(16, 29, theme)]},
),
(
"In 1981 Average hours per day by men spent on Eating and drinking ",
{"entities": [(3, 7, time),(8, 65, theme)]},
),
(
"Age of householder in 1966 ",
{"entities": [(0, 18, theme),(22, 26, time)]},
),
(
"In 1954 Other taxes of governments ",
{"entities": [(3, 7, time),(8, 34, theme)]},
),
(
"Total cost of materials of Other furniture related product manufacturing in 1958 in UK ",
{"entities": [(0, 72, theme),(76, 80, time),(84, 86, region)]},
),
(
"Difference in population density of above age 65 in 1999 ",
{"entities": [(0, 48, theme),(52, 56, time)]},
),
(
"In France annual average temperature ",
{"entities": [(3, 9, region),(10, 36, theme)]},
),
(
"In 2005 Average monthly housing cost as percentage of income ",
{"entities": [(3, 7, time),(8, 60, theme)]},
),
(
"Percent of forest area in USA by state in 1963 ",
{"entities": [(0, 22, theme),(26, 29, region),(33, 38, admin),(42, 46, time)]},
),
(
"In France in 1971 annual average precipitation ",
{"entities": [(3, 9, region),(13, 17, time),(18, 46, theme)]},
),
(
"In 1951 Elementary-secondary revenue from compensatory programs by census tract in U.S. ",
{"entities": [(3, 7, time),(8, 63, theme),(67, 79, admin),(83, 87, region)]},
),
(
"In the United States Estimated annual sales for Furniture & home furn. Stores ",
{"entities": [(3, 20, region),(21, 77, theme)]},
),
(
"By province Household income ",
{"entities": [(3, 11, admin),(12, 28, theme)]},
),
(
"Current operation of governments in 1965 by township ",
{"entities": [(0, 32, theme),(36, 40, time),(44, 52, admin)]},
),
(
"By census tract life expectancy ",
{"entities": [(3, 15, admin),(16, 31, theme)]},
),
(
"By county in 1992 Estimated annual sales for Beer, wine & liquor stores ",
{"entities": [(3, 9, admin),(13, 17, time),(18, 71, theme)]},
),
(
"Number of earthquake in US ",
{"entities": [(0, 20, theme),(24, 26, region)]},
),
(
"In the United States Utility revenue of governments ",
{"entities": [(3, 20, region),(21, 51, theme)]},
),
(
"In 2000 in UK Elementary-secondary revenue ",
{"entities": [(3, 7, time),(11, 13, region),(14, 42, theme)]},
),
(
"Average year built in 2011 ",
{"entities": [(0, 18, theme),(22, 26, time)]},
),
(
"Estimated annual sales for Building mat. & sup. dealers by township ",
{"entities": [(0, 55, theme),(59, 67, admin)]},
),
(
"By county in USA number of earthquake ",
{"entities": [(3, 9, admin),(13, 16, region),(17, 37, theme)]},
),
(
"In 1972 in USA Elementary-secondary expenditure ",
{"entities": [(3, 7, time),(11, 14, region),(15, 47, theme)]},
),
(
"By township annual average temperature ",
{"entities": [(3, 11, admin),(12, 38, theme)]},
),
(
"In Canada Annual payroll ",
{"entities": [(3, 9, region),(10, 24, theme)]},
),
(
"By state General expenditure of governments ",
{"entities": [(3, 8, admin),(9, 43, theme)]},
),
(
"In South Korea number of earthquake by township ",
{"entities": [(3, 14, region),(15, 35, theme),(39, 47, admin)]},
),
(
"Average monthly housing cost as percentage of income by state in France ",
{"entities": [(0, 52, theme),(56, 61, admin),(65, 71, region)]},
),
(
"In USA by township in 1983 difference in population density of above age 65 ",
{"entities": [(3, 6, region),(10, 18, admin),(22, 26, time),(27, 75, theme)]},
),
(
"By province in the United States annual average temperature in 1994 ",
{"entities": [(3, 11, admin),(15, 32, region),(33, 59, theme),(63, 67, time)]},
),
(
"In UK Average hours per day by women spent on Relaxing and thinking ",
{"entities": [(3, 5, region),(6, 67, theme)]},
),
(
"NSF funding for \"Catalogue\" in 2019 in UK ",
{"entities": [(0, 27, theme),(31, 35, time),(39, 41, region)]},
),
(
"In the United States by province in 1975 happiness score ",
{"entities": [(3, 20, region),(24, 32, admin),(36, 40, time),(41, 56, theme)]},
),
(
"Current spending of elementary-secondary expenditure in 2008 in U.S. ",
{"entities": [(0, 52, theme),(56, 60, time),(64, 68, region)]},
),
(
"By township annual average temperature in China ",
{"entities": [(3, 11, admin),(12, 38, theme),(42, 47, region)]},
),
(
"In 1990 Direct expenditure of governments ",
{"entities": [(3, 7, time),(8, 41, theme)]},
),
(
"Nonfamily households in South Korea ",
{"entities": [(0, 20, theme),(24, 35, region)]},
),
(
"In 1997 Elementary-secondary revenue from state sources in Canada ",
{"entities": [(3, 7, time),(8, 55, theme),(59, 65, region)]},
),
(
"In US Estimated annual sales for Beer, wine & liquor stores ",
{"entities": [(3, 5, region),(6, 59, theme)]},
),
(
"Households with householder living alone by county in UK ",
{"entities": [(0, 40, theme),(44, 50, admin),(54, 56, region)]},
),
(
"Average percent of time engaged in by menPlaying games in 1991 by census tract ",
{"entities": [(0, 54, theme),(58, 62, time),(66, 78, admin)]},
),
(
"In 2010 in the United States Annual payroll ",
{"entities": [(3, 7, time),(11, 28, region),(29, 43, theme)]},
),
(
"In 1950 Elementary-secondary revenue from parent government contributions in France ",
{"entities": [(3, 7, time),(8, 73, theme),(77, 83, region)]},
),
(
"In 1972 Estimated annual sales for Auto parts in the United States by county ",
{"entities": [(3, 7, time),(8, 45, theme),(49, 66, region),(70, 76, admin)]},
),
(
"In 1967 by province in UK Median household income ",
{"entities": [(3, 7, time),(11, 19, admin),(23, 25, region),(26, 49, theme)]},
),
(
"In U.S. by township in 2012 Elementary-secondary revenue from property taxes ",
{"entities": [(3, 7, region),(11, 19, admin),(23, 27, time),(28, 76, theme)]},
),
(
"By state in China in 1990 difference in number of people of per Walmart store ",
{"entities": [(3, 8, admin),(12, 17, region),(21, 25, time),(26, 77, theme)]},
),
(
"In 1976 Interest on debt of governments ",
{"entities": [(3, 7, time),(8, 39, theme)]},
),
(
"In 1955 in UK by county percent of households above $200k ",
{"entities": [(3, 7, time),(11, 13, region),(17, 23, admin),(24, 57, theme)]},
),
(
"In the United States by state Estimated annual sales for Mens clothing stores in 1985 ",
{"entities": [(3, 20, region),(24, 29, admin),(30, 77, theme),(81, 85, time)]},
),
(
"Number of multi-racial households in 1966 in USA ",
{"entities": [(0, 33, theme),(37, 41, time),(45, 48, region)]},
),
(
"Annual payroll in South Korea by state in 1998 ",
{"entities": [(0, 14, theme),(18, 29, region),(33, 38, admin),(42, 46, time)]},
),
(
"In South Korea Estimated annual sales for Food & beverage stores ",
{"entities": [(3, 14, region),(15, 64, theme)]},
),
(
"Average number of bedrooms of houses in the United States by state ",
{"entities": [(0, 36, theme),(40, 57, region),(61, 66, admin)]},
),
(
"In Canada Estimated annual sales for Auto & other motor veh. Dealers by province in 2001 ",
{"entities": [(3, 9, region),(10, 68, theme),(72, 80, admin),(84, 88, time)]},
),
(
"In 1958 Household income by province in USA ",
{"entities": [(3, 7, time),(8, 24, theme),(28, 36, admin),(40, 43, region)]},
),
(
"By township in 1995 Elementary-secondary revenue from parent government contributions ",
{"entities": [(3, 11, admin),(15, 19, time),(20, 85, theme)]},
),
(
"In 1963 Capital outlay of elementary-secondary expenditure by township ",
{"entities": [(3, 7, time),(8, 58, theme),(62, 70, admin)]},
),
(
"Annual average precipitation in 2019 ",
{"entities": [(0, 28, theme),(32, 36, time)]},
),
(
"In U.S. number of earthquake by state ",
{"entities": [(3, 7, region),(8, 28, theme),(32, 37, admin)]},
),
(
"In 2018 NSF funding for \"Catalogue\" by county in Canada ",
{"entities": [(3, 7, time),(8, 35, theme),(39, 45, admin),(49, 55, region)]},
),
(
"In US by state number of species ",
{"entities": [(3, 5, region),(9, 14, admin),(15, 32, theme)]},
),
(
"By township Elementary-secondary revenue from school lunch charges ",
{"entities": [(3, 11, admin),(12, 66, theme)]},
),
(
"Elementary-secondary revenue from federal sources in U.S. by township ",
{"entities": [(0, 49, theme),(53, 57, region),(61, 69, admin)]},
),
(
"By province in South Korea in 1996 Estimated annual sales for Miscellaneous store retailer ",
{"entities": [(3, 11, admin),(15, 26, region),(30, 34, time),(35, 90, theme)]},
),
(
"By province difference in number of people of people who are confirm to be infected by 2019-Nov Coronavirus in USA ",
{"entities": [(3, 11, admin),(12, 107, theme),(111, 114, region)]},
),
(
"Miscellaneous general revenue of governments by township in Canada ",
{"entities": [(0, 44, theme),(48, 56, admin),(60, 66, region)]},
),
(
"In 1999 Number of employees of Miscellaneous manufacturing ",
{"entities": [(3, 7, time),(8, 58, theme)]},
),
(
"By state in China in 1956 Number of paid employees ",
{"entities": [(3, 8, admin),(12, 17, region),(21, 25, time),(26, 50, theme)]},
),
(
"Number of paid employees in USA in 2012 ",
{"entities": [(0, 24, theme),(28, 31, region),(35, 39, time)]},
),
(
"General sales of governments in 2002 by province in USA ",
{"entities": [(0, 28, theme),(32, 36, time),(40, 48, admin),(52, 55, region)]},
),
(
"In USA Average hours per day by women spent on Travel related to education in 1973 by province ",
{"entities": [(3, 6, region),(7, 74, theme),(78, 82, time),(86, 94, admin)]},
),
(
"Average percent of time engaged in Telephone calls, mail, and e-mail in Canada ",
{"entities": [(0, 68, theme),(72, 78, region)]},
),
(
"In 1998 number of Olympic game awards ",
{"entities": [(3, 7, time),(8, 37, theme)]},
),
(
"By state in 1996 Sales, receipts, or value of shipments of firms ",
{"entities": [(3, 8, admin),(12, 16, time),(17, 64, theme)]},
),
(
"In France by census tract in 1997 Production workers average for year of Tobacco manufacturing ",
{"entities": [(3, 9, region),(13, 25, admin),(29, 33, time),(34, 94, theme)]},
),
(
"NSF funding for \"Catalogue\" in 1961 ",
{"entities": [(0, 27, theme),(31, 35, time)]},
),
(
"In Canada by census tract Average household size in 1950 ",
{"entities": [(3, 9, region),(13, 25, admin),(26, 48, theme),(52, 56, time)]},
),
(
"In China Average year built by township ",
{"entities": [(3, 8, region),(9, 27, theme),(31, 39, admin)]},
),
(
"In the United States Number of employees of Textile product mills by province ",
{"entities": [(3, 20, region),(21, 65, theme),(69, 77, admin)]},
),
(
"In 2015 by state in South Korea Average monthly housing cost ",
{"entities": [(3, 7, time),(11, 16, admin),(20, 31, region),(32, 60, theme)]},
),
(
"In 1985 Household income ",
{"entities": [(3, 7, time),(8, 24, theme)]},
),
(
"Households with female householder, no husband present, family by census tract in China ",
{"entities": [(0, 62, theme),(66, 78, admin),(82, 87, region)]},
),
(
"In 2006 Average year built by township ",
{"entities": [(3, 7, time),(8, 26, theme),(30, 38, admin)]},
),
(
"Median household income in 1973 in US by census tract ",
{"entities": [(0, 23, theme),(27, 31, time),(35, 37, region),(41, 53, admin)]},
),
(
"By county Age of householder in 1985 in France ",
{"entities": [(3, 9, admin),(10, 28, theme),(32, 36, time),(40, 46, region)]},
),
(
"Infant mortality rate in France ",
{"entities": [(0, 21, theme),(25, 31, region)]},
),
(
"Average household size by state ",
{"entities": [(0, 22, theme),(26, 31, admin)]},
),
(
"By province in France Exports value of firms in 1965 ",
{"entities": [(3, 11, admin),(15, 21, region),(22, 44, theme),(48, 52, time)]},
),
(
"By province Estimated annual sales for New car dealers ",
{"entities": [(3, 11, admin),(12, 54, theme)]},
),
(
"In 1991 Exports value of firms in France ",
{"entities": [(3, 7, time),(8, 30, theme),(34, 40, region)]},
),
(
"Percent change of people who changed the job in the past one year in USA by state in 1959 ",
{"entities": [(0, 65, theme),(69, 72, region),(76, 81, admin),(85, 89, time)]},
),
(
"Average hours per day by women spent on Helping household adults in 1967 ",
{"entities": [(0, 64, theme),(68, 72, time)]},
),
(
"By census tract energy consumption (per capita) in 1997 in France ",
{"entities": [(3, 15, admin),(16, 47, theme),(51, 55, time),(59, 65, region)]},
),
(
"In Canada Total capital expenditures of Veneer, plywood, and engineered wood product manufacturing by state in 1994 ",
{"entities": [(3, 9, region),(10, 98, theme),(102, 107, admin),(111, 115, time)]},
),
(
"Total households in the United States in 1952 ",
{"entities": [(0, 16, theme),(20, 37, region),(41, 45, time)]},
),
(
"In 1963 Elementary-secondary revenue from property taxes by state ",
{"entities": [(3, 7, time),(8, 56, theme),(60, 65, admin)]},
),
(
"In 2013 by county in China Elementary-secondary revenue from special education ",
{"entities": [(3, 7, time),(11, 17, admin),(21, 26, region),(27, 78, theme)]},
),
(
"By state annual average precipitation in UK in 2014 ",
{"entities": [(3, 8, admin),(9, 37, theme),(41, 43, region),(47, 51, time)]},
),
(
"Population density of people enrolled in Elementary school (grades 1-8) by township in UK in 1992 ",
{"entities": [(0, 71, theme),(75, 83, admin),(87, 89, region),(93, 97, time)]},
),
(
"By census tract difference in population density of separated ",
{"entities": [(3, 15, admin),(16, 61, theme)]},
),
(
"Import and export statistics in 1951 ",
{"entities": [(0, 28, theme),(32, 36, time)]},
),
(
"In 2013 unemployment rate ",
{"entities": [(3, 7, time),(8, 25, theme)]},
),
(
"In 2016 in France Average poverty level for household ",
{"entities": [(3, 7, time),(11, 17, region),(18, 53, theme)]},
),
(
"Production workers annual hours of Grain and oilseed milling by census tract ",
{"entities": [(0, 60, theme),(64, 76, admin)]},
),
(
"By census tract Sales, receipts, or value of shipments of firms in 1994 ",
{"entities": [(3, 15, admin),(16, 63, theme),(67, 71, time)]},
),
(
"In 1968 in France Salaries and wages of governments ",
{"entities": [(3, 7, time),(11, 17, region),(18, 51, theme)]},
),
(
"In 1955 Average poverty level for household ",
{"entities": [(3, 7, time),(8, 43, theme)]},
),
(
"In 1981 Number of firms ",
{"entities": [(3, 7, time),(8, 23, theme)]},
),
(
"Nonfamily households in 2007 in UK by state ",
{"entities": [(0, 20, theme),(24, 28, time),(32, 34, region),(38, 43, admin)]},
),
(
"Current spending of elementary-secondary expenditure in 1982 by province in the United States ",
{"entities": [(0, 52, theme),(56, 60, time),(64, 72, admin),(76, 93, region)]},
),
(
"Elementary-secondary revenue from state sources by township in 2008 ",
{"entities": [(0, 47, theme),(51, 59, admin),(63, 67, time)]},
),
(
"Exports value of firms in 1998 ",
{"entities": [(0, 22, theme),(26, 30, time)]},
),
(
"By township annual average temperature ",
{"entities": [(3, 11, admin),(12, 38, theme)]},
),
(
"By province Number of firms ",
{"entities": [(3, 11, admin),(12, 27, theme)]},
),
(
"In 1957 by county percent of forest area ",
{"entities": [(3, 7, time),(11, 17, admin),(18, 40, theme)]},
),
(
"In 1959 by census tract Estimated annual sales for acc. & tire store in South Korea ",
{"entities": [(3, 7, time),(11, 23, admin),(24, 68, theme),(72, 83, region)]},
),
(
"Sales, receipts, or value of shipments of firms in 1951 ",
{"entities": [(0, 47, theme),(51, 55, time)]},
),
(
"Sales, receipts, or value of shipments of firms in France ",
{"entities": [(0, 47, theme),(51, 57, region)]},
),
(
"In U.S. by province in 1961 percent of households above $200k ",
{"entities": [(3, 7, region),(11, 19, admin),(23, 27, time),(28, 61, theme)]},
),
(
"Insurance trust revenue of governments by state ",
{"entities": [(0, 38, theme),(42, 47, admin)]},
),
(
"In USA in 1976 Average hours per day by men spent on Taking class for degree, certificate, or licensure ",
{"entities": [(3, 6, region),(10, 14, time),(15, 103, theme)]},
),
(
"In China Households with female householder, no husband present, family by province in 1958 ",
{"entities": [(3, 8, region),(9, 71, theme),(75, 83, admin),(87, 91, time)]},
),
(
"In 1977 Total value of shipments and receipts for services of Apparel manufacturing by census tract in South Korea ",
{"entities": [(3, 7, time),(8, 83, theme),(87, 99, admin),(103, 114, region)]},
),
(
"In 1961 diabetes rate ",
{"entities": [(3, 7, time),(8, 21, theme)]},
),
(
"Number of paid employees in USA by province ",
{"entities": [(0, 24, theme),(28, 31, region),(35, 43, admin)]},
),
(
"In 2006 by census tract in U.S. License Taxes of governments ",
{"entities": [(3, 7, time),(11, 23, admin),(27, 31, region),(32, 60, theme)]},
),
(
"By county Average number of bedrooms of houses in 1973 ",
{"entities": [(3, 9, admin),(10, 46, theme),(50, 54, time)]},
),
(
"Elementary-secondary expenditure in the United States ",
{"entities": [(0, 32, theme),(36, 53, region)]},
),
(
"In 1990 in USA by township Elementary-secondary revenue from local sources ",
{"entities": [(3, 7, time),(11, 14, region),(18, 26, admin),(27, 74, theme)]},
),
(
"In China in 1993 homicide rate by state ",
{"entities": [(3, 8, region),(12, 16, time),(17, 30, theme),(34, 39, admin)]},
),
(
"In USA Households with one or more people under 18 years by census tract ",
{"entities": [(3, 6, region),(7, 56, theme),(60, 72, admin)]},
),
(
"Direct expenditure of governments in 1988 by county ",
{"entities": [(0, 33, theme),(37, 41, time),(45, 51, admin)]},
),
(
"In 2020 by township Number of paid employees ",
{"entities": [(3, 7, time),(11, 19, admin),(20, 44, theme)]},
),
(
"In South Korea Production workers annual hours of Cut and sew apparel manufacturing ",
{"entities": [(3, 14, region),(15, 83, theme)]},
),
(
"Percent change of White in France ",
{"entities": [(0, 23, theme),(27, 33, region)]},
),
(
"In 2010 Households with male householder, no wife present, family ",
{"entities": [(3, 7, time),(8, 65, theme)]},
),
(
"Average percent of time engaged in by womenTravel related to household activities by township in the United States in 1961 ",
{"entities": [(0, 81, theme),(85, 93, admin),(97, 114, region),(118, 122, time)]},
),
(
"In South Korea Property Taxes of governments by township ",
{"entities": [(3, 14, region),(15, 44, theme),(48, 56, admin)]},
),
(
"In 1954 in Canada by township Married-couple family ",
{"entities": [(3, 7, time),(11, 17, region),(21, 29, admin),(30, 51, theme)]},
),
(
"In France Percent of population of people who changed the job in the past one year ",
{"entities": [(3, 9, region),(10, 82, theme)]},
),
(
"Annual average precipitation by township in 1975 in UK ",
{"entities": [(0, 28, theme),(32, 40, admin),(44, 48, time),(52, 54, region)]},
),
(
"Other taxes of governments in UK ",
{"entities": [(0, 26, theme),(30, 32, region)]},
),
(
"In 1983 in USA by census tract Liquor stores expenditure of governments ",
{"entities": [(3, 7, time),(11, 14, region),(18, 30, admin),(31, 71, theme)]},
),
(
"By province Direct expenditure of governments ",
{"entities": [(3, 11, admin),(12, 45, theme)]},
),
(
"Average square footage of houses in 1982 ",
{"entities": [(0, 32, theme),(36, 40, time)]},
),
(
"Production workers annual wages of Miscellaneous manufacturing in 2020 ",
{"entities": [(0, 62, theme),(66, 70, time)]},
),
(
"Federal government expenditure (per capita) by state ",
{"entities": [(0, 43, theme),(47, 52, admin)]},
),
(
"Annual average precipitation in U.S. by township ",
{"entities": [(0, 28, theme),(32, 36, region),(40, 48, admin)]},
),
(
"In USA Average year built in 1985 by province ",
{"entities": [(3, 6, region),(7, 25, theme),(29, 33, time),(37, 45, admin)]},
),
(
"By province in China in 1990 Elementary-secondary revenue from local government ",
{"entities": [(3, 11, admin),(15, 20, region),(24, 28, time),(29, 79, theme)]},
),
(
"Assistance and subsidies of governments in USA in 1954 ",
{"entities": [(0, 39, theme),(43, 46, region),(50, 54, time)]},
),
(
"By township in U.S. Elementary-secondary revenue from special education in 1970 ",
{"entities": [(3, 11, admin),(15, 19, region),(20, 71, theme),(75, 79, time)]},
),
(
"Elementary-secondary revenue from compensatory programs in 2003 by census tract ",
{"entities": [(0, 55, theme),(59, 63, time),(67, 79, admin)]},
),
(
"Estimated annual sales for Home furnishings stores by state in UK ",
{"entities": [(0, 50, theme),(54, 59, admin),(63, 65, region)]},
),
(
"By census tract in 2004 food insecurity rate ",
{"entities": [(3, 15, admin),(19, 23, time),(24, 44, theme)]},
),
(
"In the United States Average number of bedrooms of houses in 2009 ",
{"entities": [(3, 20, region),(21, 57, theme),(61, 65, time)]},
),
(
"By state annual average temperature ",
{"entities": [(3, 8, admin),(9, 35, theme)]},
),
(
"Average percent of time engaged in by menactivities in UK by census tract ",
{"entities": [(0, 51, theme),(55, 57, region),(61, 73, admin)]},
),
(
"By county in UK Annual payroll of Computer and peripheral equipment manufacturing ",
{"entities": [(3, 9, admin),(13, 15, region),(16, 81, theme)]},
),
(
"By province annual average precipitation in 1959 ",
{"entities": [(3, 11, admin),(12, 40, theme),(44, 48, time)]},
),
(
"By census tract Family households (families) ",
{"entities": [(3, 15, admin),(16, 44, theme)]},
),
(
"Sale amounts of beer in 2016 in U.S. ",
{"entities": [(0, 20, theme),(24, 28, time),(32, 36, region)]},
),
(
"In U.S. Elementary-secondary revenue from general formula assistance in 2015 by state ",
{"entities": [(3, 7, region),(8, 68, theme),(72, 76, time),(80, 85, admin)]},
),
(
"In U.S. by state number of McDonald's ",
{"entities": [(3, 7, region),(11, 16, admin),(17, 37, theme)]},
),
(
"Estimated annual sales for Clothing & clothing accessories stores in U.S. in 2019 ",
{"entities": [(0, 65, theme),(69, 73, region),(77, 81, time)]},
),
(
"In 2011 in South Korea Estimated annual sales for Other general merch. Stores ",
{"entities": [(3, 7, time),(11, 22, region),(23, 77, theme)]},
),
(
"By census tract Households with one or more people 65 years and over in Canada ",
{"entities": [(3, 15, admin),(16, 68, theme),(72, 78, region)]},
),
(
"In USA by state Average percent of time engaged in by womenFinancial management in 2006 ",
{"entities": [(3, 6, region),(10, 15, admin),(16, 79, theme),(83, 87, time)]},
),
(
"In 1956 Household income by township ",
{"entities": [(3, 7, time),(8, 24, theme),(28, 36, admin)]},
),
(
"By township annual average temperature in 2002 ",
{"entities": [(3, 11, admin),(12, 38, theme),(42, 46, time)]},
),
(
"In 2013 Annual payroll ",
{"entities": [(3, 7, time),(8, 22, theme)]},
),
(
"In Canada Population density of widowed ",
{"entities": [(3, 9, region),(10, 39, theme)]},
),
(
"Total value of shipments and receipts for services of Dairy product manufacturing in US in 1999 ",
{"entities": [(0, 81, theme),(85, 87, region),(91, 95, time)]},
),
(
"Estimated annual sales for Food & beverage stores in 1972 ",
{"entities": [(0, 49, theme),(53, 57, time)]},
),
(
"Production workers annual hours of Other food manufacturing by county in 1962 ",
{"entities": [(0, 59, theme),(63, 69, admin),(73, 77, time)]},
),
(
"By county Capital outlay of governments in 2010 in USA ",
{"entities": [(3, 9, admin),(10, 39, theme),(43, 47, time),(51, 54, region)]},
),
(
"Household income in 1977 in UK ",
{"entities": [(0, 16, theme),(20, 24, time),(28, 30, region)]},
),
(
"Salaries and wages of governments by township ",
{"entities": [(0, 33, theme),(37, 45, admin)]},
),
(
"By county poverty rate ",
{"entities": [(3, 9, admin),(10, 22, theme)]},
),
(
"By state in 1971 in France Agriculture exports ",
{"entities": [(3, 8, admin),(12, 16, time),(20, 26, region),(27, 46, theme)]},
),
(
"In 1985 in USA by census tract NSF funding for \"Catalogue\" ",
{"entities": [(3, 7, time),(11, 14, region),(18, 30, admin),(31, 58, theme)]},
),
(
"In 2003 Average percent of time engaged in by womenAnimals and pets ",
{"entities": [(3, 7, time),(8, 67, theme)]},
),
(
"Insurance trust revenue of governments in Canada ",
{"entities": [(0, 38, theme),(42, 48, region)]},
),
(
"In 1969 in the United States by census tract GDP (nominal or ppp) ",
{"entities": [(3, 7, time),(11, 28, region),(32, 44, admin),(45, 65, theme)]},
),
(
"Elementary-secondary revenue from local government by province in 2000 in China ",
{"entities": [(0, 50, theme),(54, 62, admin),(66, 70, time),(74, 79, region)]},
),
(
"By state annual average precipitation in the United States ",
{"entities": [(3, 8, admin),(9, 37, theme),(41, 58, region)]},
),
(
"By county in South Korea in 1995 Current operation of governments ",
{"entities": [(3, 9, admin),(13, 24, region),(28, 32, time),(33, 65, theme)]},
),
(
"In 2000 by province Estimated annual sales for Warehouse clubs & supercenters ",
{"entities": [(3, 7, time),(11, 19, admin),(20, 77, theme)]},
),
(
"In 1979 in UK Households with one or more people under 18 years ",
{"entities": [(3, 7, time),(11, 13, region),(14, 63, theme)]},
),
(
"In US in 1958 diabetes rate ",
{"entities": [(3, 5, region),(9, 13, time),(14, 27, theme)]},
),
(
"Elementary-secondary revenue in 1983 ",
{"entities": [(0, 28, theme),(32, 36, time)]},
),
(
"Number of hospitals in 1993 ",
{"entities": [(0, 19, theme),(23, 27, time)]},
),
(
"In 1987 Households with householder living alone ",
{"entities": [(3, 7, time),(8, 48, theme)]},
),
(
"In France by county Direct expenditure of governments ",
{"entities": [(3, 9, region),(13, 19, admin),(20, 53, theme)]},
),
(
"In 1965 in US Production workers annual wages of Navigational, measuring, electromedical, and control instruments manufacturing ",
{"entities": [(3, 7, time),(11, 13, region),(14, 127, theme)]},
),
(
"Production workers annual wages of Other textile product mills in US by county ",
{"entities": [(0, 62, theme),(66, 68, region),(72, 78, admin)]},
),
(
"Energy consumption (per capita) in 1968 by county in South Korea ",
{"entities": [(0, 31, theme),(35, 39, time),(43, 49, admin),(53, 64, region)]},
),
(
"Married-couple family by township ",
{"entities": [(0, 21, theme),(25, 33, admin)]},
),
(
"Estimated annual sales for Food & beverage stores in the United States ",
{"entities": [(0, 49, theme),(53, 70, region)]},
),
(
"In 1979 in USA Married-couple family ",
{"entities": [(3, 7, time),(11, 14, region),(15, 36, theme)]},
),
(
"In the United States in 1997 Average percent of time engaged in by menTravel related to education by county ",
{"entities": [(3, 20, region),(24, 28, time),(29, 97, theme),(101, 107, admin)]},
),
(
"In Canada Households with female householder, no husband present, family ",
{"entities": [(3, 9, region),(10, 72, theme)]},
),
(
"In 1988 Total cost of materials of Other nonmetallic mineral product manufacturing ",
{"entities": [(3, 7, time),(8, 82, theme)]},
),
(
"By state price of land ",
{"entities": [(3, 8, admin),(9, 22, theme)]},
),
(
"In 1959 Exports value of firms ",
{"entities": [(3, 7, time),(8, 30, theme)]},
),
(
"In 1994 Married-couple family by province ",
{"entities": [(3, 7, time),(8, 29, theme),(33, 41, admin)]},
),
(
"Total capital expenditures of Sawmills and wood preservation in 1951 ",
{"entities": [(0, 60, theme),(64, 68, time)]},
),
(
"Elementary-secondary revenue from general formula assistance in 1951 in France by township ",
{"entities": [(0, 60, theme),(64, 68, time),(72, 78, region),(82, 90, admin)]},
),
(
"By census tract Sales and Gross Receipts Taxes of governments ",
{"entities": [(3, 15, admin),(16, 61, theme)]},
),
(
"Median household income in USA ",
{"entities": [(0, 23, theme),(27, 30, region)]},
),
(
"Number of firms in 1994 by state ",
{"entities": [(0, 15, theme),(19, 23, time),(27, 32, admin)]},
),
(
"In U.S. in 1965 by census tract Capital outlay of elementary-secondary expenditure ",
{"entities": [(3, 7, region),(11, 15, time),(19, 31, admin),(32, 82, theme)]},
),
(
"By state Estimated annual sales for Motor vehicle & parts Dealers in 1994 ",
{"entities": [(3, 8, admin),(9, 65, theme),(69, 73, time)]},
),
(
"By state General sales of governments in 1982 ",
{"entities": [(3, 8, admin),(9, 37, theme),(41, 45, time)]},
),
(
"In 1980 by province in US Total value of shipments and receipts for services of Machinery manufacturing ",
{"entities": [(3, 7, time),(11, 19, admin),(23, 25, region),(26, 103, theme)]},
),
(
"In 2003 by state in France Estimated annual sales for Miscellaneous store retailer ",
{"entities": [(3, 7, time),(11, 16, admin),(20, 26, region),(27, 82, theme)]},
),
(
"Difference in number of people of above age 65 in 2004 ",
{"entities": [(0, 46, theme),(50, 54, time)]},
),
(
"By province in China Total value of shipments and receipts for services of Seafood product preparation and packaging in 2008 ",
{"entities": [(3, 11, admin),(15, 20, region),(21, 116, theme),(120, 124, time)]},
),
(
"In 1965 in France Average percent of time engaged in Working by province ",
{"entities": [(3, 7, time),(11, 17, region),(18, 60, theme),(64, 72, admin)]},
),
(
"In 1998 in France by census tract Annual payroll ",
{"entities": [(3, 7, time),(11, 17, region),(21, 33, admin),(34, 48, theme)]},
),
(
"Difference in number of people of Jewish in 1960 by township ",
{"entities": [(0, 40, theme),(44, 48, time),(52, 60, admin)]},
),
(
"In UK number of people of black or African American ",
{"entities": [(3, 5, region),(6, 51, theme)]},
),
(
"In US number of earthquake ",
{"entities": [(3, 5, region),(6, 26, theme)]},
),
(
"In France Average hours per day by men spent on Attending class by county in 1977 ",
{"entities": [(3, 9, region),(10, 63, theme),(67, 73, admin),(77, 81, time)]},
),
(
"Measles incidence in France in 1962 by county ",
{"entities": [(0, 17, theme),(21, 27, region),(31, 35, time),(39, 45, admin)]},
),
(
"By county annual average temperature in 1960 in UK ",
{"entities": [(3, 9, admin),(10, 36, theme),(40, 44, time),(48, 50, region)]},
),
(
"In Canada Households with female householder, no husband present, family ",
{"entities": [(3, 9, region),(10, 72, theme)]},
),
(
"In US in 2005 by census tract difference in population density of people enrolled in Elementary school (grades 1-8) ",
{"entities": [(3, 5, region),(9, 13, time),(17, 29, admin),(30, 115, theme)]},
),
(
"Annual average temperature by township ",
{"entities": [(0, 26, theme),(30, 38, admin)]},
),
(
"Intergovernmental expenditure of governments in South Korea ",
{"entities": [(0, 44, theme),(48, 59, region)]},
),
(
"Difference in population density of per Walmart store in 1973 ",
{"entities": [(0, 53, theme),(57, 61, time)]},
),
(
"Interest on general debt of governments by census tract in South Korea in 1950 ",
{"entities": [(0, 39, theme),(43, 55, admin),(59, 70, region),(74, 78, time)]},
),
(
"Number of fire points in 1957 ",
{"entities": [(0, 21, theme),(25, 29, time)]},
),
(
"In the United States in 1955 Total households ",
{"entities": [(3, 20, region),(24, 28, time),(29, 45, theme)]},
),
(
"In 1957 number of fire points ",
{"entities": [(3, 7, time),(8, 29, theme)]},
),
(
"Nonfamily households by county ",
{"entities": [(0, 20, theme),(24, 30, admin)]},
),
(
"Number of firms by census tract ",
{"entities": [(0, 15, theme),(19, 31, admin)]},
),
(
"In UK in 1987 number of people of people enrolled in Elementary school (grades 1-8) ",
{"entities": [(3, 5, region),(9, 13, time),(14, 83, theme)]},
),
(
"In 1968 Estimated annual sales for Clothing & clothing accessories stores in UK by county ",
{"entities": [(3, 7, time),(8, 73, theme),(77, 79, region),(83, 89, admin)]},
),
(
"Annual average temperature by state ",
{"entities": [(0, 26, theme),(30, 35, admin)]},
),
(
"Insurance benefits and repayments of governments in China in 1986 ",
{"entities": [(0, 48, theme),(52, 57, region),(61, 65, time)]},
),
(
"In the United States in 2012 Agriculture exports ",
{"entities": [(3, 20, region),(24, 28, time),(29, 48, theme)]},
),
(
"By province in 1966 Average percent of time engaged in by menLeisure and sports in the United States ",
{"entities": [(3, 11, admin),(15, 19, time),(20, 79, theme),(83, 100, region)]},
),
(
"In 2010 in Canada by census tract Liquor stores expenditure of governments ",
{"entities": [(3, 7, time),(11, 17, region),(21, 33, admin),(34, 74, theme)]},
),
(
"In USA Production workers average for year of Commercial and service industry machinery manufacturing ",
{"entities": [(3, 6, region),(7, 101, theme)]},
),
(
"In 1980 in USA by county Average monthly housing cost as percentage of income ",
{"entities": [(3, 7, time),(11, 14, region),(18, 24, admin),(25, 77, theme)]},
),
(
"Elementary-secondary revenue from local sources in 2002 by province ",
{"entities": [(0, 47, theme),(51, 55, time),(59, 67, admin)]},
),
(
"In UK number of academic articles published in 1976 ",
{"entities": [(3, 5, region),(6, 43, theme),(47, 51, time)]},
),
(
"Annual payroll in UK in 1994 ",
{"entities": [(0, 14, theme),(18, 20, region),(24, 28, time)]},
),
(
"In 2018 Elementary-secondary revenue from transportation programs by province ",
{"entities": [(3, 7, time),(8, 65, theme),(69, 77, admin)]},
),
(
"In 2016 Production workers annual hours of Other chemical product and preparation manufacturing by province in China ",
{"entities": [(3, 7, time),(8, 95, theme),(99, 107, admin),(111, 116, region)]},
),
(
"In the United States Family households (families) ",
{"entities": [(3, 20, region),(21, 49, theme)]},
),
(
"In 1994 Production workers average for year of Paint, coating, and adhesive manufacturing ",
{"entities": [(3, 7, time),(8, 89, theme)]},
),
(
"Annual average precipitation in the United States in 1992 by township ",
{"entities": [(0, 28, theme),(32, 49, region),(53, 57, time),(61, 69, admin)]},
),
(
"Estimated annual sales for Womens clothing stores in China in 1991 by county ",
{"entities": [(0, 49, theme),(53, 58, region),(62, 66, time),(70, 76, admin)]},
),
(
"In South Korea Annual payroll of Lime and gypsum product manufacturing in 1959 by township ",
{"entities": [(3, 14, region),(15, 70, theme),(74, 78, time),(82, 90, admin)]},
),
(
"Sale amounts of beer in Canada by township ",
{"entities": [(0, 20, theme),(24, 30, region),(34, 42, admin)]},
),
(
"Number of fire points in China ",
{"entities": [(0, 21, theme),(25, 30, region)]},
),
(
"Elementary-secondary revenue from local government by census tract in 1960 ",
{"entities": [(0, 50, theme),(54, 66, admin),(70, 74, time)]},
),
(
"Estimated annual sales for total (excl. motor vehicle & parts & gasoline stations) in the United States in 1979 ",
{"entities": [(0, 82, theme),(86, 103, region),(107, 111, time)]},
),
(
"By province Households with one or more people under 18 years in 1953 ",
{"entities": [(3, 11, admin),(12, 61, theme),(65, 69, time)]},
),
(
"In 1998 Average year built by census tract in USA ",
{"entities": [(3, 7, time),(8, 26, theme),(30, 42, admin),(46, 49, region)]},
),
(
"By township Estimated annual sales for Auto parts in France in 1981 ",
{"entities": [(3, 11, admin),(12, 49, theme),(53, 59, region),(63, 67, time)]},
),
(
"In 2004 Direct expenditure of governments ",
{"entities": [(3, 7, time),(8, 41, theme)]},
),
(
"By census tract in South Korea Total capital expenditures of Agriculture, construction, and mining machinery manufacturing ",
{"entities": [(3, 15, admin),(19, 30, region),(31, 122, theme)]},
),
(
"By province Number of employees of Wood product manufacturing ",
{"entities": [(3, 11, admin),(12, 61, theme)]},
),
(
"Poverty rate in Canada in 2014 by county ",
{"entities": [(0, 12, theme),(16, 22, region),(26, 30, time),(34, 40, admin)]},
),
(
"Average percent of time engaged in by womenIndoor and outdoor maintenance, building, and cleanup activities in 2012 by census tract in UK ",
{"entities": [(0, 107, theme),(111, 115, time),(119, 131, admin),(135, 137, region)]},
),
(
"By state Estimated annual sales for Food & beverage stores ",
{"entities": [(3, 8, admin),(9, 58, theme)]},
),
(
"By province Number of employees of Aerospace product and parts manufacturing in 1957 in USA ",
{"entities": [(3, 11, admin),(12, 76, theme),(80, 84, time),(88, 91, region)]},
),
(
"In US by township Elementary-secondary revenue from federal sources in 1998 ",
{"entities": [(3, 5, region),(9, 17, admin),(18, 67, theme),(71, 75, time)]},
),
(
"In U.S. Estimated annual sales for Food & beverage stores in 2015 ",
{"entities": [(3, 7, region),(8, 57, theme),(61, 65, time)]},
),
(
"Freedom index in 1971 ",
{"entities": [(0, 13, theme),(17, 21, time)]},
),
(
"In 1961 Total cost of materials of Beverage and tobacco product manufacturing ",
{"entities": [(3, 7, time),(8, 77, theme)]},
),
(
"Average percent of time engaged in by womenHelping household children with Homework in France by province ",
{"entities": [(0, 83, theme),(87, 93, region),(97, 105, admin)]},
),
(
"Nonfamily households by province in the United States ",
{"entities": [(0, 20, theme),(24, 32, admin),(36, 53, region)]},
),
(
"Annual average temperature in U.S. by township ",
{"entities": [(0, 26, theme),(30, 34, region),(38, 46, admin)]},
),
(
"In 1971 import and export statistics in Canada ",
{"entities": [(3, 7, time),(8, 36, theme),(40, 46, region)]},
),
(
"In 1954 Percent of population of males 15 years and over ",
{"entities": [(3, 7, time),(8, 56, theme)]},
),
(
"In 1963 Exports value of firms by state in UK ",
{"entities": [(3, 7, time),(8, 30, theme),(34, 39, admin),(43, 45, region)]},
),
(
"In 1984 in U.S. Average hours per day by women spent on Caring for and helping household children ",
{"entities": [(3, 7, time),(11, 15, region),(16, 97, theme)]},
),
(
"By state in the United States Estimated annual sales for Clothing & clothing accessories stores ",
{"entities": [(3, 8, admin),(12, 29, region),(30, 95, theme)]},
),
(
"Age of householder in 1958 in U.S. ",
{"entities": [(0, 18, theme),(22, 26, time),(30, 34, region)]},
),
(
"In France Current operation of governments by province in 1995 ",
{"entities": [(3, 9, region),(10, 42, theme),(46, 54, admin),(58, 62, time)]},
),
(
"By township Intergovernmental revenue of governments in US in 1971 ",
{"entities": [(3, 11, admin),(12, 52, theme),(56, 58, region),(62, 66, time)]},
),
(
"By census tract difference in population density of now married, except separated in France ",
{"entities": [(3, 15, admin),(16, 81, theme),(85, 91, region)]},
),
(
"In 1993 Average hours per day by women spent on Attending household children events by state in UK ",
{"entities": [(3, 7, time),(8, 83, theme),(87, 92, admin),(96, 98, region)]},
),
(
"Average percent of time engaged in Socializing and communicating by province ",
{"entities": [(0, 64, theme),(68, 76, admin)]},
),
(
"Difference in number of people of males 15 years and over in 2008 in US ",
{"entities": [(0, 57, theme),(61, 65, time),(69, 71, region)]},
),
(
"In US in 1965 number of Olympic game awards ",
{"entities": [(3, 5, region),(9, 13, time),(14, 43, theme)]},
),
(
"By province divorce rate in 2000 ",
{"entities": [(3, 11, admin),(12, 24, theme),(28, 32, time)]},
),
(
"Total capital expenditures of Fruit and vegetable preserving and specialty food manufacturing in 1995 in UK ",
{"entities": [(0, 93, theme),(97, 101, time),(105, 107, region)]},
),
(
"Average hours per day spent on Civic obligations and participation in 1950 in Canada ",
{"entities": [(0, 66, theme),(70, 74, time),(78, 84, region)]},
),
(
"Average family size by state ",
{"entities": [(0, 19, theme),(23, 28, admin)]},
),
(
"By county in 1965 Sales, receipts, or value of shipments of firms ",
{"entities": [(3, 9, admin),(13, 17, time),(18, 65, theme)]},
),
(
"By state Estimated annual sales for Grocery stores ",
{"entities": [(3, 8, admin),(9, 50, theme)]},
),
(
"By county Average household size ",
{"entities": [(3, 9, admin),(10, 32, theme)]},
),
(
"In USA in 2001 by county Elementary-secondary revenue from compensatory programs ",
{"entities": [(3, 6, region),(10, 14, time),(18, 24, admin),(25, 80, theme)]},
),
(
"In UK by county in 1999 Elementary-secondary revenue from vocational programs ",
{"entities": [(3, 5, region),(9, 15, admin),(19, 23, time),(24, 77, theme)]},
),
(
"Elementary-secondary expenditure in the United States ",
{"entities": [(0, 32, theme),(36, 53, region)]},
),
(
"In 2006 helicobacter pylori rate ",
{"entities": [(3, 7, time),(8, 32, theme)]},
),
(
"In UK in 1960 Production workers annual hours of Machinery manufacturing by township ",
{"entities": [(3, 5, region),(9, 13, time),(14, 72, theme),(76, 84, admin)]},
),
(
"In 1984 Estimated annual sales for total (excl. motor vehicle & parts & gasoline stations) by county ",
{"entities": [(3, 7, time),(8, 90, theme),(94, 100, admin)]},
),
(
"By province in 1978 in US number of patent per capita ",
{"entities": [(3, 11, admin),(15, 19, time),(23, 25, region),(26, 53, theme)]},
),
(
"By state number of earthquake in China in 2008 ",
{"entities": [(3, 8, admin),(9, 29, theme),(33, 38, region),(42, 46, time)]},
),
(
"Estimated annual sales for Motor vehicle & parts Dealers in Canada ",
{"entities": [(0, 56, theme),(60, 66, region)]},
),
(
"In 1971 in US Production workers annual hours of Cement and concrete product manufacturing by state ",
{"entities": [(3, 7, time),(11, 13, region),(14, 90, theme),(94, 99, admin)]},
),
(
"By province Estimated annual sales for Sporting goods hobby, musical instrument, & book stores ",
{"entities": [(3, 11, admin),(12, 94, theme)]},
),
(
"In 1987 number of earthquake ",
{"entities": [(3, 7, time),(8, 28, theme)]},
),
(
"Capital outlay of governments by census tract in 1998 ",
{"entities": [(0, 29, theme),(33, 45, admin),(49, 53, time)]},
),
(
"Household income in 1957 by county ",
{"entities": [(0, 16, theme),(20, 24, time),(28, 34, admin)]},
),
(
"Percent of households above $200k in 1956 ",
{"entities": [(0, 33, theme),(37, 41, time)]},
),
(
"By census tract crime rate ",
{"entities": [(3, 15, admin),(16, 26, theme)]},
),
(
"By state General expenditure of governments in 1952 ",
{"entities": [(3, 8, admin),(9, 43, theme),(47, 51, time)]},
),
(
"In France Average percent of time engaged in by womenHome maintenance, repair, decoration, and construction (not done by self) ",
{"entities": [(3, 9, region),(10, 126, theme)]},
),
(
"In the United States Insurance trust expenditure of governments by province in 1987 ",
{"entities": [(3, 20, region),(21, 63, theme),(67, 75, admin),(79, 83, time)]},
),
(
"In 1986 Production workers average for year of Household and institutional furniture and kitchen cabinet manufacturing in France by county ",
{"entities": [(3, 7, time),(8, 118, theme),(122, 128, region),(132, 138, admin)]},
),
(
"Percent of population of people enrolled in Kindergarten in the United States ",
{"entities": [(0, 56, theme),(60, 77, region)]},
),
(
"In 2009 in Canada difference in number of people of who believe climate change ",
{"entities": [(3, 7, time),(11, 17, region),(18, 78, theme)]},
),
(
"By province average age ",
{"entities": [(3, 11, admin),(12, 23, theme)]},
),
(
"By state Households with male householder, no wife present, family ",
{"entities": [(3, 8, admin),(9, 66, theme)]},
),
(
"Nonfamily households in 2011 in France ",
{"entities": [(0, 20, theme),(24, 28, time),(32, 38, region)]},
),
(
"In 1982 Households with householder living alone by census tract ",
{"entities": [(3, 7, time),(8, 48, theme),(52, 64, admin)]},
),
(
"In 1984 Average family size ",
{"entities": [(3, 7, time),(8, 27, theme)]},
),
(
"In 2011 by state Average hours per day by men spent on Home maintenance, repair, decoration, and construction (not done by self) in US ",
{"entities": [(3, 7, time),(11, 16, admin),(17, 128, theme),(132, 134, region)]},
),
(
"By province difference in population density of dentist in 1994 in Canada ",
{"entities": [(3, 11, admin),(12, 55, theme),(59, 63, time),(67, 73, region)]},
),
(
"Average percent of time engaged in by womenCaring for household adults by state ",
{"entities": [(0, 70, theme),(74, 79, admin)]},
),
(
"By state in U.S. in 1987 Median household income ",
{"entities": [(3, 8, admin),(12, 16, region),(20, 24, time),(25, 48, theme)]},
),
(
"Average square footage of houses in USA in 1968 by township ",
{"entities": [(0, 32, theme),(36, 39, region),(43, 47, time),(51, 59, admin)]},
),
(
"By province in 1997 in China Households with one or more people 65 years and over ",
{"entities": [(3, 11, admin),(15, 19, time),(23, 28, region),(29, 81, theme)]},
),
(
"Average hours per day by women spent on Household and personal messages by township in the United States ",
{"entities": [(0, 71, theme),(75, 83, admin),(87, 104, region)]},
),
(
"Percent of farmland in China by township in 1981 ",
{"entities": [(0, 19, theme),(23, 28, region),(32, 40, admin),(44, 48, time)]},
),
(
"In Canada Number of paid employees ",
{"entities": [(3, 9, region),(10, 34, theme)]},
),
(
"Capital outlay of elementary-secondary expenditure in the United States by state ",
{"entities": [(0, 50, theme),(54, 71, region),(75, 80, admin)]},
),
(
"In UK Exports value of firms ",
{"entities": [(3, 5, region),(6, 28, theme)]},
),
(
"In 1968 Number of firms by province ",
{"entities": [(3, 7, time),(8, 23, theme),(27, 35, admin)]},
),
(
"In U.S. in 1990 difference in number of people of who believe climate change ",
{"entities": [(3, 7, region),(11, 15, time),(16, 76, theme)]},
),
(
"Average number of bedrooms of houses by county ",
{"entities": [(0, 36, theme),(40, 46, admin)]},
),
(
"NSF funding for \"Catalogue\" in 1995 ",
{"entities": [(0, 27, theme),(31, 35, time)]},
),
(
"In 2000 in U.S. annual average precipitation by state ",
{"entities": [(3, 7, time),(11, 15, region),(16, 44, theme),(48, 53, admin)]},
),
(
"By township in 1986 difference in population density of Native Hawaiian and Other Pacific Islander ",
{"entities": [(3, 11, admin),(15, 19, time),(20, 98, theme)]},
),
(
"In China in 2003 number of earthquake ",
{"entities": [(3, 8, region),(12, 16, time),(17, 37, theme)]},
),
(
"Race diversity index by county in UK ",
{"entities": [(0, 20, theme),(24, 30, admin),(34, 36, region)]},
),
(
"By province in 1967 difference in population density of separated in France ",
{"entities": [(3, 11, admin),(15, 19, time),(20, 65, theme),(69, 75, region)]},
),
(
"By state in USA Population density of separated ",
{"entities": [(3, 8, admin),(12, 15, region),(16, 47, theme)]},
),
(
"Households with one or more people 65 years and over in 1967 in the United States by county ",
{"entities": [(0, 52, theme),(56, 60, time),(64, 81, region),(85, 91, admin)]},
),
(
"By state in France Estimated annual sales for total (excl. motor vehicle & parts & gasoline stations) ",
{"entities": [(3, 8, admin),(12, 18, region),(19, 101, theme)]},
),
(
"By census tract in 1959 food insecurity rate in Canada ",
{"entities": [(3, 15, admin),(19, 23, time),(24, 44, theme),(48, 54, region)]},
),
(
"Age of householder in 1980 ",
{"entities": [(0, 18, theme),(22, 26, time)]},
),
(
"By county in France in 1990 Average percent of time engaged in by womenSocializing, relaxing, and leisure ",
{"entities": [(3, 9, admin),(13, 19, region),(23, 27, time),(28, 105, theme)]},
),
(
"Average poverty level for household in 1954 ",
{"entities": [(0, 35, theme),(39, 43, time)]},
),
(
"In U.S. Current spending of elementary-secondary expenditure by state in 1960 ",
{"entities": [(3, 7, region),(8, 60, theme),(64, 69, admin),(73, 77, time)]},
),
(
"By province in France Estimated annual sales for Department stores in 1974 ",
{"entities": [(3, 11, admin),(15, 21, region),(22, 66, theme),(70, 74, time)]},
),
(
"Average square footage of houses in France in 2009 ",
{"entities": [(0, 32, theme),(36, 42, region),(46, 50, time)]},
),
(
"In 1962 number of earthquake ",
{"entities": [(3, 7, time),(8, 28, theme)]},
),
(
"In France by census tract in 1985 Direct expenditure of governments ",
{"entities": [(3, 9, region),(13, 25, admin),(29, 33, time),(34, 67, theme)]},
),
(
"Annual average precipitation by county in 1973 in the United States ",
{"entities": [(0, 28, theme),(32, 38, admin),(42, 46, time),(50, 67, region)]},
),
(
"Elementary-secondary revenue from vocational programs in France ",
{"entities": [(0, 53, theme),(57, 63, region)]},
),
(
"Average year built in 2011 in South Korea ",
{"entities": [(0, 18, theme),(22, 26, time),(30, 41, region)]},
),
(
"By state Households with male householder, no wife present, family ",
{"entities": [(3, 8, admin),(9, 66, theme)]},
),
(
"In 1960 Sales, receipts, or value of shipments of firms in Canada ",
{"entities": [(3, 7, time),(8, 55, theme),(59, 65, region)]},
),
(
"Nonfamily households in USA ",
{"entities": [(0, 20, theme),(24, 27, region)]},
),
(
"Households with one or more people under 18 years by county ",
{"entities": [(0, 49, theme),(53, 59, admin)]},
),
(
"In U.S. Estimated annual sales for Furniture stores ",
{"entities": [(3, 7, region),(8, 51, theme)]},
),
(
"In France Elementary-secondary revenue from special education by township ",
{"entities": [(3, 9, region),(10, 61, theme),(65, 73, admin)]},
),
(
"In 1972 by state in France Average hours per day spent on Laundry ",
{"entities": [(3, 7, time),(11, 16, admin),(20, 26, region),(27, 65, theme)]},
),
(
"By state Elementary-secondary revenue from special education in 1957 ",
{"entities": [(3, 8, admin),(9, 60, theme),(64, 68, time)]},
),
(
"Exports value of firms in U.S. in 1956 by census tract ",
{"entities": [(0, 22, theme),(26, 30, region),(34, 38, time),(42, 54, admin)]},
),
(
"Population density of people enrolled in Elementary school (grades 1-8) in 1974 in South Korea ",
{"entities": [(0, 71, theme),(75, 79, time),(83, 94, region)]},
),
(
"Estimated annual sales for Food services & drinking places in China by state ",
{"entities": [(0, 58, theme),(62, 67, region),(71, 76, admin)]},
),
(
"In 1994 in the United States Average hours per day by men spent on Government services ",
{"entities": [(3, 7, time),(11, 28, region),(29, 86, theme)]},
),
(
"By province Average hours per day by women spent on Sports, exercise, and recreation in 1966 ",
{"entities": [(3, 11, admin),(12, 84, theme),(88, 92, time)]},
),
(
"In 1953 General revenue of governments by province in the United States ",
{"entities": [(3, 7, time),(8, 38, theme),(42, 50, admin),(54, 71, region)]},
),
(
"By province in 1989 in UK Elementary-secondary revenue ",
{"entities": [(3, 11, admin),(15, 19, time),(23, 25, region),(26, 54, theme)]},
),
(
"Number of fixed residential broadband providers in 2001 by census tract ",
{"entities": [(0, 47, theme),(51, 55, time),(59, 71, admin)]},
),
(
"By township annual average temperature ",
{"entities": [(3, 11, admin),(12, 38, theme)]},
),
(
"In the United States Utility expenditure of governments ",
{"entities": [(3, 20, region),(21, 55, theme)]},
),
(
"In 1967 by township in South Korea Total Taxes of governments ",
{"entities": [(3, 7, time),(11, 19, admin),(23, 34, region),(35, 61, theme)]},
),
(
"Household income in France ",
{"entities": [(0, 16, theme),(20, 26, region)]},
),
(
"In UK Production workers annual wages of Manufacturing ",
{"entities": [(3, 5, region),(6, 54, theme)]},
),
(
"In 1955 by state in US Average hours per day by men spent on Storing interior household items, including food ",
{"entities": [(3, 7, time),(11, 16, admin),(20, 22, region),(23, 109, theme)]},
),
(
"In 1963 Household income in US ",
{"entities": [(3, 7, time),(8, 24, theme),(28, 30, region)]},
),
(
"Annual average precipitation by census tract ",
{"entities": [(0, 28, theme),(32, 44, admin)]},
),
(
"In 2017 in US by township Average year built ",
{"entities": [(3, 7, time),(11, 13, region),(17, 25, admin),(26, 44, theme)]},
),
(
"In 1967 Elementary-secondary revenue from other state aid ",
{"entities": [(3, 7, time),(8, 57, theme)]},
),
(
"In 1972 Elementary-secondary revenue from transportation programs in US ",
{"entities": [(3, 7, time),(8, 65, theme),(69, 71, region)]},
),
(
"Population density of males 15 years and over in China in 1977 by county ",
{"entities": [(0, 45, theme),(49, 54, region),(58, 62, time),(66, 72, admin)]},
),
(
"In US Households with female householder, no husband present, family by census tract in 1958 ",
{"entities": [(3, 5, region),(6, 68, theme),(72, 84, admin),(88, 92, time)]},
),
(
"In 1989 Average year built ",
{"entities": [(3, 7, time),(8, 26, theme)]},
),
(
"By state Sales, receipts, or value of shipments of firms in UK ",
{"entities": [(3, 8, admin),(9, 56, theme),(60, 62, region)]},
),
(
"Number of paid employees in 1989 in China by county ",
{"entities": [(0, 24, theme),(28, 32, time),(36, 41, region),(45, 51, admin)]},
),
(
"In UK Estimated annual sales for Auto parts ",
{"entities": [(3, 5, region),(6, 43, theme)]},
),
(
"Average percent of time engaged in by menHousehold and personal messages by province in the United States in 2020 ",
{"entities": [(0, 72, theme),(76, 84, admin),(88, 105, region),(109, 113, time)]},
),
(
"In US by county annual average temperature ",
{"entities": [(3, 5, region),(9, 15, admin),(16, 42, theme)]},
),
(
"In 1972 Estimated annual sales for Home furnishings stores by census tract in the United States ",
{"entities": [(3, 7, time),(8, 58, theme),(62, 74, admin),(78, 95, region)]},
),
(
"In South Korea in 1997 percent of households above $200k by county ",
{"entities": [(3, 14, region),(18, 22, time),(23, 56, theme),(60, 66, admin)]},
),
(
"Number of people of people who are alumni of OSU by province ",
{"entities": [(0, 48, theme),(52, 60, admin)]},
),
(
"Number of employees of Commercial and service industry machinery manufacturing in 2007 by census tract in the United States ",
{"entities": [(0, 78, theme),(82, 86, time),(90, 102, admin),(106, 123, region)]},
),
(
"Average poverty level for household in Canada ",
{"entities": [(0, 35, theme),(39, 45, region)]},
),
(
"By township in 1950 in U.S. Intergovernmental expenditure of governments ",
{"entities": [(3, 11, admin),(15, 19, time),(23, 27, region),(28, 72, theme)]},
),
(
"In 2007 in Canada Average hours per day by men spent on Travel related to education by county ",
{"entities": [(3, 7, time),(11, 17, region),(18, 83, theme),(87, 93, admin)]},
),
(
"In Canada in 1969 Estimated annual sales for Womens clothing stores ",
{"entities": [(3, 9, region),(13, 17, time),(18, 67, theme)]},
),
(
"In China by county in 1971 Insurance trust revenue of governments ",
{"entities": [(3, 8, region),(12, 18, admin),(22, 26, time),(27, 65, theme)]},
),
(
"In 1957 by province Direct expenditure of governments in Canada ",
{"entities": [(3, 7, time),(11, 19, admin),(20, 53, theme),(57, 63, region)]},
),
(
"By county in US in 1951 Property Taxes of governments ",
{"entities": [(3, 9, admin),(13, 15, region),(19, 23, time),(24, 53, theme)]},
),
(
"In 1997 Average hours per day by men spent on Travel related to caring for and helping nonhousehold membership ",
{"entities": [(3, 7, time),(8, 110, theme)]},
),
(
"In U.S. in 1969 Interest on debt of governments ",
{"entities": [(3, 7, region),(11, 15, time),(16, 47, theme)]},
),
(
"Elementary-secondary expenditure in 2014 by township ",
{"entities": [(0, 32, theme),(36, 40, time),(44, 52, admin)]},
),
(
"Infant mortality rate in 1972 ",
{"entities": [(0, 21, theme),(25, 29, time)]},
),
(
"In 1960 by census tract annual average temperature in China ",
{"entities": [(3, 7, time),(11, 23, admin),(24, 50, theme),(54, 59, region)]},
),
(
"General sales of governments in China ",
{"entities": [(0, 28, theme),(32, 37, region)]},
),
(
"In 1983 by county Households with householder living alone in US ",
{"entities": [(3, 7, time),(11, 17, admin),(18, 58, theme),(62, 64, region)]},
),
(
"Percent change of divorced by county in Canada ",
{"entities": [(0, 26, theme),(30, 36, admin),(40, 46, region)]},
),
(
"In Canada by census tract in 1965 Utility expenditure of governments ",
{"entities": [(3, 9, region),(13, 25, admin),(29, 33, time),(34, 68, theme)]},
),
(
"In France Production workers average for year of Audio and video equipment manufacturing in 1995 ",
{"entities": [(3, 9, region),(10, 88, theme),(92, 96, time)]},
),
(
"Estimated annual sales for Motor vehicle & parts Dealers in U.S. ",
{"entities": [(0, 56, theme),(60, 64, region)]},
),
(
"By county in China in 2008 Total cost of materials of Navigational, measuring, electromedical, and control instruments manufacturing ",
{"entities": [(3, 9, admin),(13, 18, region),(22, 26, time),(27, 132, theme)]},
),
(
"In 2002 by county Cash and security holdings of governments in South Korea ",
{"entities": [(3, 7, time),(11, 17, admin),(18, 59, theme),(63, 74, region)]},
),
(
"Average hours per day by women spent on Volunteering (organizational and civic activities) by county ",
{"entities": [(0, 90, theme),(94, 100, admin)]},
),
(
"Average number of bedrooms of houses in France by province in 2018 ",
{"entities": [(0, 36, theme),(40, 46, region),(50, 58, admin),(62, 66, time)]},
),
(
"By county GDP (nominal or ppp) in Canada ",
{"entities": [(3, 9, admin),(10, 30, theme),(34, 40, region)]},
),
(
"In USA by county Average percent of time engaged in Taking class for degree, certificate, or licensure ",
{"entities": [(3, 6, region),(10, 16, admin),(17, 102, theme)]},
),
(
"In US Average percent of time engaged in by menReading for personal interest ",
{"entities": [(3, 5, region),(6, 76, theme)]},
),
(
"By province happiness score ",
{"entities": [(3, 11, admin),(12, 27, theme)]},
),
(
"In the United States in 1998 average price for honey per pound ",
{"entities": [(3, 20, region),(24, 28, time),(29, 62, theme)]},
),
(
"In the United States Average percent of time engaged in Travel related to organizational, civic, and religious activities ",
{"entities": [(3, 20, region),(21, 121, theme)]},
),
(
"In 1973 Household income in the United States ",
{"entities": [(3, 7, time),(8, 24, theme),(28, 45, region)]},
),
(
"Married-couple family in US in 2017 ",
{"entities": [(0, 21, theme),(25, 27, region),(31, 35, time)]},
),
(
"In 1976 by township in Canada Average number of bedrooms of houses ",
{"entities": [(3, 7, time),(11, 19, admin),(23, 29, region),(30, 66, theme)]},
),
(
"In the United States Estimated annual sales for acc. & tire store by province ",
{"entities": [(3, 20, region),(21, 65, theme),(69, 77, admin)]},
),
(
"Average hours per day by women spent on Physical care for household children in 2008 ",
{"entities": [(0, 76, theme),(80, 84, time)]},
),
(
"Number of fixed residential broadband providers in UK ",
{"entities": [(0, 47, theme),(51, 53, region)]},
),
(
"In 1989 in France Total cost of materials of Architectural and structural metals manufacturing ",
{"entities": [(3, 7, time),(11, 17, region),(18, 94, theme)]},
),
(
"In 1979 in South Korea Estimated annual sales for General merchandise stores ",
{"entities": [(3, 7, time),(11, 22, region),(23, 76, theme)]},
),
(
"Exports value of firms by state ",
{"entities": [(0, 22, theme),(26, 31, admin)]},
),
(
"Number of paid employees in 1972 in China ",
{"entities": [(0, 24, theme),(28, 32, time),(36, 41, region)]},
),
(
"In 1952 by state in the United States Estimated annual sales for Auto parts ",
{"entities": [(3, 7, time),(11, 16, admin),(20, 37, region),(38, 75, theme)]},
),
(
"By province Production workers average for year of Apparel manufacturing ",
{"entities": [(3, 11, admin),(12, 72, theme)]},
),
(
"Annual average precipitation in 1987 ",
{"entities": [(0, 28, theme),(32, 36, time)]},
),
(
"In U.S. by province Elementary-secondary revenue from vocational programs ",
{"entities": [(3, 7, region),(11, 19, admin),(20, 73, theme)]},
),
(
"By county in 1952 Percent of population of males 15 years and over ",
{"entities": [(3, 9, admin),(13, 17, time),(18, 66, theme)]},
),
(
"In 1952 freedom index in US ",
{"entities": [(3, 7, time),(8, 21, theme),(25, 27, region)]},
),
(
"In 1988 Production workers average for year of Manufacturing by province ",
{"entities": [(3, 7, time),(8, 60, theme),(64, 72, admin)]},
),
(
"In 2008 in USA Total households by province ",
{"entities": [(3, 7, time),(11, 14, region),(15, 31, theme),(35, 43, admin)]},
),
(
"By township number of earthquake ",
{"entities": [(3, 11, admin),(12, 32, theme)]},
),
(
"Percent change of Native Hawaiian and Other Pacific Islander by province in France ",
{"entities": [(0, 60, theme),(64, 72, admin),(76, 82, region)]},
),
(
"By county Number of paid employees in the United States ",
{"entities": [(3, 9, admin),(10, 34, theme),(38, 55, region)]},
),
(
"In USA in 2001 Family households with own children of the householder under 18 years ",
{"entities": [(3, 6, region),(10, 14, time),(15, 84, theme)]},
),
(
"Number of earthquake in U.S. in 1978 by county ",
{"entities": [(0, 20, theme),(24, 28, region),(32, 36, time),(40, 46, admin)]},
),
(
"Exports value of firms in France ",
{"entities": [(0, 22, theme),(26, 32, region)]},
),
(
"By county Estimated annual sales for Department stores in 1958 ",
{"entities": [(3, 9, admin),(10, 54, theme),(58, 62, time)]},
),
(
"Average hours per day spent on Household and personal e-mail and messages in 1957 by province ",
{"entities": [(0, 73, theme),(77, 81, time),(85, 93, admin)]},
),
(
"In 1990 Average square footage of houses ",
{"entities": [(3, 7, time),(8, 40, theme)]},
),
(
"In Canada Sales, receipts, or value of shipments of firms by county ",
{"entities": [(3, 9, region),(10, 57, theme),(61, 67, admin)]},
),
(
"In US Annual payroll of Grain and oilseed milling by province ",
{"entities": [(3, 5, region),(6, 49, theme),(53, 61, admin)]},
),
(
"By province in 2004 GDP (nominal or ppp) per capita in UK ",
{"entities": [(3, 11, admin),(15, 19, time),(20, 51, theme),(55, 57, region)]},
),
(
"By township Elementary-secondary revenue from school lunch charges in Canada in 2001 ",
{"entities": [(3, 11, admin),(12, 66, theme),(70, 76, region),(80, 84, time)]},
),
(
"Number of paid employees in China ",
{"entities": [(0, 24, theme),(28, 33, region)]},
),
(
"In South Korea Corporate income tax of governments ",
{"entities": [(3, 14, region),(15, 50, theme)]},
),
(
"In France Elementary-secondary revenue from general formula assistance ",
{"entities": [(3, 9, region),(10, 70, theme)]},
),
(
"In 2006 number of people of people enrolled in College or graduate school ",
{"entities": [(3, 7, time),(8, 73, theme)]},
),
(
"In US in 1987 Average hours per day spent on Telephone calls, mail, and e-mail ",
{"entities": [(3, 5, region),(9, 13, time),(14, 78, theme)]},
),
(
"Elementary-secondary revenue from property taxes in 1988 ",
{"entities": [(0, 48, theme),(52, 56, time)]},
),
(
"Number of fire points by township in 1987 in UK ",
{"entities": [(0, 21, theme),(25, 33, admin),(37, 41, time),(45, 47, region)]},
),
(
"By county Total revenue of governments in 2003 in U.S. ",
{"entities": [(3, 9, admin),(10, 38, theme),(42, 46, time),(50, 54, region)]},
),
(
"In China in 1986 by state annual average temperature ",
{"entities": [(3, 8, region),(12, 16, time),(20, 25, admin),(26, 52, theme)]},
),
(
"By census tract Average family size ",
{"entities": [(3, 15, admin),(16, 35, theme)]},
),
(
"Annual payroll in UK by state in 1957 ",
{"entities": [(0, 14, theme),(18, 20, region),(24, 29, admin),(33, 37, time)]},
),
(
"In 1962 Age of householder ",
{"entities": [(3, 7, time),(8, 26, theme)]},
),
(
"Diabetes rate in 1986 by census tract in France ",
{"entities": [(0, 13, theme),(17, 21, time),(25, 37, admin),(41, 47, region)]},
),
(
"In 1973 in US by state Number of paid employees ",
{"entities": [(3, 7, time),(11, 13, region),(17, 22, admin),(23, 47, theme)]},
),
(
"Utility revenue of governments in 1952 in the United States ",
{"entities": [(0, 30, theme),(34, 38, time),(42, 59, region)]},
),
(
"By census tract Estimated annual sales for Motor vehicle & parts Dealers in UK ",
{"entities": [(3, 15, admin),(16, 72, theme),(76, 78, region)]},
),
(
"In 1959 in South Korea Elementary-secondary revenue from local sources ",
{"entities": [(3, 7, time),(11, 22, region),(23, 70, theme)]},
),
(
"By state Percent of population of Asian in 1965 ",
{"entities": [(3, 8, admin),(9, 39, theme),(43, 47, time)]},
),
(
"By township in 2002 in South Korea Households with female householder, no husband present, family ",
{"entities": [(3, 11, admin),(15, 19, time),(23, 34, region),(35, 97, theme)]},
),
(
"Difference in number of people of Catholic in South Korea ",
{"entities": [(0, 42, theme),(46, 57, region)]},
),
(
"By census tract Percent of population of people who are alumni of OSU ",
{"entities": [(3, 15, admin),(16, 69, theme)]},
),
(
"Number of people of per Walmart store by province in USA in 1965 ",
{"entities": [(0, 37, theme),(41, 49, admin),(53, 56, region),(60, 64, time)]},
),
(
"Total capital expenditures of Metalworking machinery manufacturing in 2014 by census tract in China ",
{"entities": [(0, 66, theme),(70, 74, time),(78, 90, admin),(94, 99, region)]},
),
(
"In France Estimated annual sales for Grocery stores ",
{"entities": [(3, 9, region),(10, 51, theme)]},
),
(
"In UK by census tract Married-couple family in 1976 ",
{"entities": [(3, 5, region),(9, 21, admin),(22, 43, theme),(47, 51, time)]},
),
(
"Elementary-secondary revenue by census tract ",
{"entities": [(0, 28, theme),(32, 44, admin)]},
),
(
"Difference in number of people of People whose native language is Russian by census tract ",
{"entities": [(0, 73, theme),(77, 89, admin)]},
),
(
"By province in UK in 2005 percent of houses with annual income of $300,000 and over ",
{"entities": [(3, 11, admin),(15, 17, region),(21, 25, time),(26, 83, theme)]},
),
(
"By township Annual payroll in 1973 ",
{"entities": [(3, 11, admin),(12, 26, theme),(30, 34, time)]},
),
(
"By township in China number of fire points in 2001 ",
{"entities": [(3, 11, admin),(15, 20, region),(21, 42, theme),(46, 50, time)]},
),
(
"By township Estimated annual sales for Home furnishings stores ",
{"entities": [(3, 11, admin),(12, 62, theme)]},
),
(
"Households with male householder, no wife present, family by state ",
{"entities": [(0, 57, theme),(61, 66, admin)]},
),
(
"In 1967 by province in France Household income ",
{"entities": [(3, 7, time),(11, 19, admin),(23, 29, region),(30, 46, theme)]},
),
(
"In USA by county Production workers annual wages of Foundries ",
{"entities": [(3, 6, region),(10, 16, admin),(17, 61, theme)]},
),
(
"In USA number of Olympic game awards ",
{"entities": [(3, 6, region),(7, 36, theme)]},
),
(
"In US in 2005 Average percent of time engaged in by menWorking and work-related activities by township ",
{"entities": [(3, 5, region),(9, 13, time),(14, 90, theme),(94, 102, admin)]},
),
(
"In UK by county in 2012 annual average precipitation ",
{"entities": [(3, 5, region),(9, 15, admin),(19, 23, time),(24, 52, theme)]},
),
(
"In 1995 Households with householder living alone ",
{"entities": [(3, 7, time),(8, 48, theme)]},
),
(
"Interest on general debt of governments in Canada in 1959 by township ",
{"entities": [(0, 39, theme),(43, 49, region),(53, 57, time),(61, 69, admin)]},
),
(
"In 2009 in the United States Average poverty level for household by county ",
{"entities": [(3, 7, time),(11, 28, region),(29, 64, theme),(68, 74, admin)]},
),
(
"Current charge of governments by province ",
{"entities": [(0, 29, theme),(33, 41, admin)]},
),
(
"In 1978 in Canada Estimated annual sales for Building mat. & sup. dealers ",
{"entities": [(3, 7, time),(11, 17, region),(18, 73, theme)]},
),
(
"In 1972 Average hours per day spent on Household and personal mail and messages ",
{"entities": [(3, 7, time),(8, 79, theme)]},
),
(
"By province in 2016 in Canada Elementary-secondary revenue from federal sources ",
{"entities": [(3, 11, admin),(15, 19, time),(23, 29, region),(30, 79, theme)]},
),
(
"In 1956 Average family size in USA ",
{"entities": [(3, 7, time),(8, 27, theme),(31, 34, region)]},
),
(
"Number of schools by county in 2002 in China ",
{"entities": [(0, 17, theme),(21, 27, admin),(31, 35, time),(39, 44, region)]},
),
(
"Elementary-secondary revenue from school lunch charges by census tract ",
{"entities": [(0, 54, theme),(58, 70, admin)]},
),
(
"In France number of fire points ",
{"entities": [(3, 9, region),(10, 31, theme)]},
),
(
"In USA percent of forest area in 1972 by township ",
{"entities": [(3, 6, region),(7, 29, theme),(33, 37, time),(41, 49, admin)]},
),
(
"Elementary-secondary revenue from general formula assistance in South Korea by county ",
{"entities": [(0, 60, theme),(64, 75, region),(79, 85, admin)]},
),
(
"In US annual average precipitation ",
{"entities": [(3, 5, region),(6, 34, theme)]},
),
(
"By state in China Households with one or more people under 18 years ",
{"entities": [(3, 8, admin),(12, 17, region),(18, 67, theme)]},
),
(
"Production workers annual hours of Boiler, tank, and shipping container manufacturing in 1955 ",
{"entities": [(0, 85, theme),(89, 93, time)]},
),
(
"In Canada by township diabetes rate in 2004 ",
{"entities": [(3, 9, region),(13, 21, admin),(22, 35, theme),(39, 43, time)]},
),
(
"Percent change of people enrolled in Nursery school, people enrolled in preschool in 2009 ",
{"entities": [(0, 81, theme),(85, 89, time)]},
),
(
"By state economic growth rate ",
{"entities": [(3, 8, admin),(9, 29, theme)]},
),
(
"By census tract Annual payroll ",
{"entities": [(3, 15, admin),(16, 30, theme)]},
),
(
"By county Elementary-secondary revenue from compensatory programs ",
{"entities": [(3, 9, admin),(10, 65, theme)]},
),
(
"Number of fire points in 2002 ",
{"entities": [(0, 21, theme),(25, 29, time)]},
),
(
"Estimated annual sales for Family clothing stores in South Korea by census tract in 1969 ",
{"entities": [(0, 49, theme),(53, 64, region),(68, 80, admin),(84, 88, time)]},
),
(
"Average percent of time engaged in by menPhysical care for household children by census tract in 2008 in France ",
{"entities": [(0, 77, theme),(81, 93, admin),(97, 101, time),(105, 111, region)]},
),
(
"In 2017 Population density of per Walmart store in USA by state ",
{"entities": [(3, 7, time),(8, 47, theme),(51, 54, region),(58, 63, admin)]},
),
(
"Exports value of firms by county in 1984 ",
{"entities": [(0, 22, theme),(26, 32, admin),(36, 40, time)]},
),
(
"Annual average precipitation by county ",
{"entities": [(0, 28, theme),(32, 38, admin)]},
),
(
"Annual average temperature in U.S. by census tract in 2018 ",
{"entities": [(0, 26, theme),(30, 34, region),(38, 50, admin),(54, 58, time)]},
),
(
"In 1971 social vulnerability index ",
{"entities": [(3, 7, time),(8, 34, theme)]},
),
(
"By county Average hours per day by women spent on Relaxing and thinking ",
{"entities": [(3, 9, admin),(10, 71, theme)]},
),
(
"By province annual average temperature ",
{"entities": [(3, 11, admin),(12, 38, theme)]},
),
(
"In US Total Taxes of governments ",
{"entities": [(3, 5, region),(6, 32, theme)]},
),
(
"In South Korea median rent price ",
{"entities": [(3, 14, region),(15, 32, theme)]},
),
(
"By state Direct expenditure of governments ",
{"entities": [(3, 8, admin),(9, 42, theme)]},
),
(
"In Canada by province Elementary-secondary revenue from local sources in 2004 ",
{"entities": [(3, 9, region),(13, 21, admin),(22, 69, theme),(73, 77, time)]},
),
(
"In Canada by county in 2013 License Taxes of governments ",
{"entities": [(3, 9, region),(13, 19, admin),(23, 27, time),(28, 56, theme)]},
),
(
"By township Production workers annual wages of Grain and oilseed milling in 1989 in Canada ",
{"entities": [(3, 11, admin),(12, 72, theme),(76, 80, time),(84, 90, region)]},
),
(
"By township Households with householder living alone in US ",
{"entities": [(3, 11, admin),(12, 52, theme),(56, 58, region)]},
),
(
"In 2011 in Canada by province Population density of people enrolled in Kindergarten ",
{"entities": [(3, 7, time),(11, 17, region),(21, 29, admin),(30, 83, theme)]},
),
(
"In US in 1993 number of fire points by census tract ",
{"entities": [(3, 5, region),(9, 13, time),(14, 35, theme),(39, 51, admin)]},
),
(
"In South Korea in 1967 Estimated annual sales for Motor vehicle & parts Dealers by county ",
{"entities": [(3, 14, region),(18, 22, time),(23, 79, theme),(83, 89, admin)]},
),
(
"In US availability of safe drinking water by county in 1953 ",
{"entities": [(3, 5, region),(6, 41, theme),(45, 51, admin),(55, 59, time)]},
),
(
"Production workers annual hours of Apparel knitting mills in 1978 by census tract in Canada ",
{"entities": [(0, 57, theme),(61, 65, time),(69, 81, admin),(85, 91, region)]},
),
(
"Number of employees of Other general purpose machinery manufacturing in France ",
{"entities": [(0, 68, theme),(72, 78, region)]},
),
(
"By census tract in 1962 gross domestic income (nominal or ppp) in US ",
{"entities": [(3, 15, admin),(19, 23, time),(24, 62, theme),(66, 68, region)]},
),
(
"Elementary-secondary revenue in USA ",
{"entities": [(0, 28, theme),(32, 35, region)]},
),
(
"Estimated annual sales for General merchandise stores in South Korea ",
{"entities": [(0, 53, theme),(57, 68, region)]},
),
(
"In 1960 by township in US Nonfamily households ",
{"entities": [(3, 7, time),(11, 19, admin),(23, 25, region),(26, 46, theme)]},
),
(
"In Canada in 1994 Number of firms by census tract ",
{"entities": [(3, 9, region),(13, 17, time),(18, 33, theme),(37, 49, admin)]},
),
(
"By province Annual payroll of Converted paper product manufacturing in the United States in 1966 ",
{"entities": [(3, 11, admin),(12, 67, theme),(71, 88, region),(92, 96, time)]},
),
(
"By county annual average precipitation in UK ",
{"entities": [(3, 9, admin),(10, 38, theme),(42, 44, region)]},
),
(
"Number of fire points by township in USA ",
{"entities": [(0, 21, theme),(25, 33, admin),(37, 40, region)]},
),
(
"By county in the United States Elementary-secondary revenue from federal sources in 2015 ",
{"entities": [(3, 9, admin),(13, 30, region),(31, 80, theme),(84, 88, time)]},
),
(
"In U.S. Total cost of materials of Food manufacturing ",
{"entities": [(3, 7, region),(8, 53, theme)]},
),
(
"In 2008 number of earthquake ",
{"entities": [(3, 7, time),(8, 28, theme)]},
),
(
"By province difference in number of people of Asian ",
{"entities": [(3, 11, admin),(12, 51, theme)]},
),
(
"By state in France difference in number of people of Hispanic or Latino Origin in 2008 ",
{"entities": [(3, 8, admin),(12, 18, region),(19, 78, theme),(82, 86, time)]},
),
(
"In 2015 by province in Canada Estimated annual sales for Furniture & home furn. Stores ",
{"entities": [(3, 7, time),(11, 19, admin),(23, 29, region),(30, 86, theme)]},
),
(
"Difference in population density of All Race in China ",
{"entities": [(0, 44, theme),(48, 53, region)]},
),
(
"In 2005 in China by state Number of firms ",
{"entities": [(3, 7, time),(11, 16, region),(20, 25, admin),(26, 41, theme)]},
),
(
"In China by province availability of safe drinking water ",
{"entities": [(3, 8, region),(12, 20, admin),(21, 56, theme)]},
),
(
"By township Estimated annual sales for total (excl. gasoline stations) ",
{"entities": [(3, 11, admin),(12, 70, theme)]},
),
(
"In US Elementary-secondary revenue from vocational programs ",
{"entities": [(3, 5, region),(6, 59, theme)]},
),
(
"In 1987 in France Average percent of time engaged in by womenactivities by county ",
{"entities": [(3, 7, time),(11, 17, region),(18, 71, theme),(75, 81, admin)]},
),
(
"By census tract in U.S. Estimated annual sales for Clothing & clothing accessories stores ",
{"entities": [(3, 15, admin),(19, 23, region),(24, 89, theme)]},
),
(
"Elementary-secondary revenue from transportation programs by county in Canada ",
{"entities": [(0, 57, theme),(61, 67, admin),(71, 77, region)]},
),
(
"In 1969 Average percent of time engaged in Consumer goods purchases ",
{"entities": [(3, 7, time),(8, 67, theme)]},
),
(
"In 2008 in UK annual average precipitation ",
{"entities": [(3, 7, time),(11, 13, region),(14, 42, theme)]},
),
(
"In 2001 Average family size ",
{"entities": [(3, 7, time),(8, 27, theme)]},
),
(
"In China difference in number of people of Jewish in 1954 ",
{"entities": [(3, 8, region),(9, 49, theme),(53, 57, time)]},
),
(
"In Canada in 1972 by township Production workers annual hours of Computer and electronic product manufacturing ",
{"entities": [(3, 9, region),(13, 17, time),(21, 29, admin),(30, 110, theme)]},
),
(
"Annual payroll of Tobacco manufacturing in Canada in 1998 by census tract ",
{"entities": [(0, 39, theme),(43, 49, region),(53, 57, time),(61, 73, admin)]},
),
(
"License taxes of governments by province ",
{"entities": [(0, 28, theme),(32, 40, admin)]},
),
(
"In 1994 Average hours per day spent on Household services ",
{"entities": [(3, 7, time),(8, 57, theme)]},
),
(
"In 2020 Average percent of time engaged in by womenTravel related to education by county in Canada ",
{"entities": [(3, 7, time),(8, 78, theme),(82, 88, admin),(92, 98, region)]},
),
(
"In 1989 in USA by state Elementary-secondary revenue from general formula assistance ",
{"entities": [(3, 7, time),(11, 14, region),(18, 23, admin),(24, 84, theme)]},
),
(
"By province Percent change of White ",
{"entities": [(3, 11, admin),(12, 35, theme)]},
),
(
"By province Income Taxes of governments in Canada in 1962 ",
{"entities": [(3, 11, admin),(12, 39, theme),(43, 49, region),(53, 57, time)]},
),
(
"Elementary-secondary revenue from state sources by township in France in 1964 ",
{"entities": [(0, 47, theme),(51, 59, admin),(63, 69, region),(73, 77, time)]},
),
(
"Current spending of elementary-secondary expenditure in 1969 ",
{"entities": [(0, 52, theme),(56, 60, time)]},
),
(
"In 1969 sale amounts of beer by county in UK ",
{"entities": [(3, 7, time),(8, 28, theme),(32, 38, admin),(42, 44, region)]},
),
(
"In 1994 price of land in USA ",
{"entities": [(3, 7, time),(8, 21, theme),(25, 28, region)]},
),
(
"In France Production workers average for year of Soap, cleaning compound, and toilet preparation manufacturing ",
{"entities": [(3, 9, region),(10, 110, theme)]},
),
(
"Capital outlay of governments in USA in 2011 by township ",
{"entities": [(0, 29, theme),(33, 36, region),(40, 44, time),(48, 56, admin)]},
),
(
"In the United States by township suicide rate ",
{"entities": [(3, 20, region),(24, 32, admin),(33, 45, theme)]},
),
(
"In U.S. import and export statistics ",
{"entities": [(3, 7, region),(8, 36, theme)]},
),
(
"Elementary-secondary revenue from local government in the United States by census tract ",
{"entities": [(0, 50, theme),(54, 71, region),(75, 87, admin)]},
),
(
"In 1985 by county Estimated annual sales for All oth. gen. merch. Stores in Canada ",
{"entities": [(3, 7, time),(11, 17, admin),(18, 72, theme),(76, 82, region)]},
),
(
"Percent of population of people who are alumni of OSU in 1994 in US by census tract ",
{"entities": [(0, 53, theme),(57, 61, time),(65, 67, region),(71, 83, admin)]},
),
(
"In USA number of libraries ",
{"entities": [(3, 6, region),(7, 26, theme)]},
),
(
"By county in South Korea Elementary-secondary revenue from local government in 1971 ",
{"entities": [(3, 9, admin),(13, 24, region),(25, 75, theme),(79, 83, time)]},
),
(
"In UK annual average temperature by township in 1999 ",
{"entities": [(3, 5, region),(6, 32, theme),(36, 44, admin),(48, 52, time)]},
),
(
"In U.S. Estimated annual sales for Pharmacies & drug stores ",
{"entities": [(3, 7, region),(8, 59, theme)]},
),
(
"By province Age of householder in the United States ",
{"entities": [(3, 11, admin),(12, 30, theme),(34, 51, region)]},
),
(
"In France by census tract Number of paid employees ",
{"entities": [(3, 9, region),(13, 25, admin),(26, 50, theme)]},
),
(
"Estimated annual sales for Womens clothing stores by county ",
{"entities": [(0, 49, theme),(53, 59, admin)]},
),
(
"In US annual average temperature ",
{"entities": [(3, 5, region),(6, 32, theme)]},
),
(
"In 1956 helicobacter pylori rate ",
{"entities": [(3, 7, time),(8, 32, theme)]},
),
(
"By township in 1999 in France import and export statistics ",
{"entities": [(3, 11, admin),(15, 19, time),(23, 29, region),(30, 58, theme)]},
),
(
"By county in 1982 in South Korea Number of paid employees ",
{"entities": [(3, 9, admin),(13, 17, time),(21, 32, region),(33, 57, theme)]},
),
(
"In 2014 by county Average hours per day by women spent on Sports, exercise, and recreation ",
{"entities": [(3, 7, time),(11, 17, admin),(18, 90, theme)]},
),
(
"By state in China Capital outlay of elementary-secondary expenditure ",
{"entities": [(3, 8, admin),(12, 17, region),(18, 68, theme)]},
),
(
"Number of patent per capita in China by state in 1979 ",
{"entities": [(0, 27, theme),(31, 36, region),(40, 45, admin),(49, 53, time)]},
),
(
"In 1952 in South Korea Estimated annual sales for Elect. shopping & m/o houses ",
{"entities": [(3, 7, time),(11, 22, region),(23, 78, theme)]},
),
(
"Import and export statistics in South Korea ",
{"entities": [(0, 28, theme),(32, 43, region)]},
),
(
"In 1975 in US Percent change of people enrolled in College or graduate school by state ",
{"entities": [(3, 7, time),(11, 13, region),(14, 77, theme),(81, 86, admin)]},
),
(
"Percent change of Catholic in US ",
{"entities": [(0, 26, theme),(30, 32, region)]},
),
(
"By township in USA in 2019 General revenue of governments ",
{"entities": [(3, 11, admin),(15, 18, region),(22, 26, time),(27, 57, theme)]},
),
(
"In 1959 Elementary-secondary revenue from local sources ",
{"entities": [(3, 7, time),(8, 55, theme)]},
),
(
"In US by township Percent change of people enrolled in Elementary school (grades 1-8) in 2007 ",
{"entities": [(3, 5, region),(9, 17, admin),(18, 85, theme),(89, 93, time)]},
),
(
"Number of fire points in 1966 by province ",
{"entities": [(0, 21, theme),(25, 29, time),(33, 41, admin)]},
),
(
"In the United States Elementary-secondary revenue from property taxes ",
{"entities": [(3, 20, region),(21, 69, theme)]},
),
(
"In US Average hours per day by women spent on Travel related to organizational, civic, and religious activities in 1988 ",
{"entities": [(3, 5, region),(6, 111, theme),(115, 119, time)]},
),
(
"In the United States Direct expenditure of governments ",
{"entities": [(3, 20, region),(21, 54, theme)]},
),
(
"By province Estimated annual sales for General merchandise stores in 2010 ",
{"entities": [(3, 11, admin),(12, 65, theme),(69, 73, time)]},
),
(
"Suicide rate in 2019 ",
{"entities": [(0, 12, theme),(16, 20, time)]},
),
(
"In France Production workers average for year of Machine shops; turned product; and screw, nut, and bolt manscrewufacturing ",
{"entities": [(3, 9, region),(10, 123, theme)]},
),
(
"In USA in 1984 by state Household income ",
{"entities": [(3, 6, region),(10, 14, time),(18, 23, admin),(24, 40, theme)]},
),
(
"By township annual average precipitation ",
{"entities": [(3, 11, admin),(12, 40, theme)]},
),
(
"In China Average family size ",
{"entities": [(3, 8, region),(9, 28, theme)]},
),
(
"In USA by census tract Direct expenditure of governments in 1989 ",
{"entities": [(3, 6, region),(10, 22, admin),(23, 56, theme),(60, 64, time)]},
),
(
"Estimated annual sales for acc. & tire store in the United States in 2010 ",
{"entities": [(0, 44, theme),(48, 65, region),(69, 73, time)]},
),
(
"CO2 emission (per capita) by county in USA ",
{"entities": [(0, 25, theme),(29, 35, admin),(39, 42, region)]},
),
(
"Average monthly housing cost as percentage of income by state in China ",
{"entities": [(0, 52, theme),(56, 61, admin),(65, 70, region)]},
),
(
"Average hours per day spent on Playing games in 1960 ",
{"entities": [(0, 44, theme),(48, 52, time)]},
),
(
"In 2014 Estimated annual sales for Motor vehicle & parts Dealers ",
{"entities": [(3, 7, time),(8, 64, theme)]},
),
(
"Estimated annual sales for General merchandise stores in 1994 ",
{"entities": [(0, 53, theme),(57, 61, time)]},
),
(
"Age of householder in USA by province ",
{"entities": [(0, 18, theme),(22, 25, region),(29, 37, admin)]},
),
(
"Diabetes rate by county ",
{"entities": [(0, 13, theme),(17, 23, admin)]},
),
(
"By province in USA Elementary-secondary revenue in 1986 ",
{"entities": [(3, 11, admin),(15, 18, region),(19, 47, theme),(51, 55, time)]},
),
(
"Number of people of Muslim by township ",
{"entities": [(0, 26, theme),(30, 38, admin)]},
),
(
"By census tract in 1957 Average number of bedrooms of houses in USA ",
{"entities": [(3, 15, admin),(19, 23, time),(24, 60, theme),(64, 67, region)]},
),
(
"In 1985 number of earthquake ",
{"entities": [(3, 7, time),(8, 28, theme)]},
),
(
"In Canada by state in 1952 annual average precipitation ",
{"entities": [(3, 9, region),(13, 18, admin),(22, 26, time),(27, 55, theme)]},
),
(
"In 2007 Estimated annual sales for Department stores ",
{"entities": [(3, 7, time),(8, 52, theme)]},
),
(
"In the United States in 1979 Total households by state ",
{"entities": [(3, 20, region),(24, 28, time),(29, 45, theme),(49, 54, admin)]},
),
(
"In 1998 Estimated annual sales for total (excl. motor vehicle & parts) ",
{"entities": [(3, 7, time),(8, 70, theme)]},
),
(
"In South Korea Total expenditure of governments ",
{"entities": [(3, 14, region),(15, 47, theme)]},
),
(
"By province Intergovernmental expenditure of governments in 2002 ",
{"entities": [(3, 11, admin),(12, 56, theme),(60, 64, time)]},
),
(
"Annual average temperature in 1978 ",
{"entities": [(0, 26, theme),(30, 34, time)]},
),
(
"In US annual average precipitation ",
{"entities": [(3, 5, region),(6, 34, theme)]},
),
(
"Median household income by county ",
{"entities": [(0, 23, theme),(27, 33, admin)]},
),
(
"In 2014 difference in number of people of death among children under 5 due to pediatric cancer ",
{"entities": [(3, 7, time),(8, 94, theme)]},
),
(
"In South Korea by county Sales, receipts, or value of shipments of firms ",
{"entities": [(3, 14, region),(18, 24, admin),(25, 72, theme)]},
),
(
"In South Korea by township Total cost of materials of Soap, cleaning compound, and toilet preparation manufacturing ",
{"entities": [(3, 14, region),(18, 26, admin),(27, 115, theme)]},
),
(
"Insurance trust expenditure of governments in 1995 ",
{"entities": [(0, 42, theme),(46, 50, time)]},
),
(
"Insurance trust expenditure of governments in 2006 ",
{"entities": [(0, 42, theme),(46, 50, time)]},
),
(
"In USA in 1967 Estimated annual sales for Electronics & appliance stores by township ",
{"entities": [(3, 6, region),(10, 14, time),(15, 72, theme),(76, 84, admin)]},
),
(
"Total expenditure of governments by township in U.S. in 1999 ",
{"entities": [(0, 32, theme),(36, 44, admin),(48, 52, region),(56, 60, time)]},
),
(
"In South Korea General expenditure of governments by census tract ",
{"entities": [(3, 14, region),(15, 49, theme),(53, 65, admin)]},
),
(
"In 1985 by county Average family size ",
{"entities": [(3, 7, time),(11, 17, admin),(18, 37, theme)]},
),
(
"By township number of people of separated in U.S. in 1994 ",
{"entities": [(3, 11, admin),(12, 41, theme),(45, 49, region),(53, 57, time)]},
),
(
"In 1960 by state Elementary-secondary revenue from property taxes ",
{"entities": [(3, 7, time),(11, 16, admin),(17, 65, theme)]},
),
(
"Total value of shipments and receipts for services of Footwear manufacturing in 1956 ",
{"entities": [(0, 76, theme),(80, 84, time)]},
),
(
"Average percent of time engaged in by womenHome maintenance, repair, decoration, and construction (not done by self) in US ",
{"entities": [(0, 116, theme),(120, 122, region)]},
),
(
"Estimated annual sales for Furniture & home furn. Stores in 1991 in US ",
{"entities": [(0, 56, theme),(60, 64, time),(68, 70, region)]},
),
(
"In 1974 by census tract rate of male in USA ",
{"entities": [(3, 7, time),(11, 23, admin),(24, 36, theme),(40, 43, region)]},
),
(
"By census tract CO2 emission (per capita) ",
{"entities": [(3, 15, admin),(16, 41, theme)]},
),
(
"In 1958 federal government expenditure (per capita) by county in U.S. ",
{"entities": [(3, 7, time),(8, 51, theme),(55, 61, admin),(65, 69, region)]},
),
(
"In the United States Elementary-secondary revenue from transportation programs in 1972 ",
{"entities": [(3, 20, region),(21, 78, theme),(82, 86, time)]},
),
(
"Households with householder living alone by county in 1987 ",
{"entities": [(0, 40, theme),(44, 50, admin),(54, 58, time)]},
),
(
"By census tract Sales, receipts, or value of shipments of firms ",
{"entities": [(3, 15, admin),(16, 63, theme)]},
),
(
"Estimated annual sales for Furniture stores in US by census tract ",
{"entities": [(0, 43, theme),(47, 49, region),(53, 65, admin)]},
),
(
"Exports value of firms in US ",
{"entities": [(0, 22, theme),(26, 28, region)]},
),
(
"In the United States Miscellaneous general revenue of governments by census tract in 2020 ",
{"entities": [(3, 20, region),(21, 65, theme),(69, 81, admin),(85, 89, time)]},
),
(
"Production workers average for year of Sawmills and wood preservation in U.S. by county ",
{"entities": [(0, 69, theme),(73, 77, region),(81, 87, admin)]},
),
(
"By state Miscellaneous general revenue of governments in China ",
{"entities": [(3, 8, admin),(9, 53, theme),(57, 62, region)]},
),
(
"Average family size in 2011 in South Korea by state ",
{"entities": [(0, 19, theme),(23, 27, time),(31, 42, region),(46, 51, admin)]},
),
(
"In South Korea by county Sales and Gross Receipts Taxes of governments in 2002 ",
{"entities": [(3, 14, region),(18, 24, admin),(25, 70, theme),(74, 78, time)]},
),
(
"In 2008 in U.S. Elementary-secondary revenue from general formula assistance ",
{"entities": [(3, 7, time),(11, 15, region),(16, 76, theme)]},
),
(
"In Canada number of fire points in 2008 by province ",
{"entities": [(3, 9, region),(10, 31, theme),(35, 39, time),(43, 51, admin)]},
),
(
"By county in 2020 annual average temperature in USA ",
{"entities": [(3, 9, admin),(13, 17, time),(18, 44, theme),(48, 51, region)]},
),
(
"In 1979 in France Current operation of governments ",
{"entities": [(3, 7, time),(11, 17, region),(18, 50, theme)]},
),
(
"By state in USA number of pedestrian accidents in 1990 ",
{"entities": [(3, 8, admin),(12, 15, region),(16, 46, theme),(50, 54, time)]},
),
(
"In Canada Taxes of governments by township in 1997 ",
{"entities": [(3, 9, region),(10, 30, theme),(34, 42, admin),(46, 50, time)]},
),
(
"Average hours per day by women spent on Travel related to organizational, civic, and religious activities in U.S. in 2010 by census tract ",
{"entities": [(0, 105, theme),(109, 113, region),(117, 121, time),(125, 137, admin)]},
),
(
"In U.S. in 2006 Estimated annual sales for General merchandise stores ",
{"entities": [(3, 7, region),(11, 15, time),(16, 69, theme)]},
),
(
"By state annual average precipitation ",
{"entities": [(3, 8, admin),(9, 37, theme)]},
),
(
"In France Elementary-secondary revenue from federal sources by county in 1995 ",
{"entities": [(3, 9, region),(10, 59, theme),(63, 69, admin),(73, 77, time)]},
),
(
"In France Elementary-secondary revenue from property taxes ",
{"entities": [(3, 9, region),(10, 58, theme)]},
),
(
"Number of fire points in US by township ",
{"entities": [(0, 21, theme),(25, 27, region),(31, 39, admin)]},
),
(
"In South Korea Production workers annual hours of Forging and stamping,Cutlery and handtool manufacturing by census tract ",
{"entities": [(3, 14, region),(15, 105, theme),(109, 121, admin)]},
),
(
"Number of paid employees in USA in 1990 ",
{"entities": [(0, 24, theme),(28, 31, region),(35, 39, time)]},
),
(
"In USA in 2007 Average percent of time engaged in by womenWorking by census tract ",
{"entities": [(3, 6, region),(10, 14, time),(15, 65, theme),(69, 81, admin)]},
),
(
"Estimated annual sales for Mens clothing stores in the United States in 2010 ",
{"entities": [(0, 47, theme),(51, 68, region),(72, 76, time)]},
),
(
"In Canada Number of employees of Motor vehicle parts manufacturing ",
{"entities": [(3, 9, region),(10, 66, theme)]},
),
(
"Flu incidence by state ",
{"entities": [(0, 13, theme),(17, 22, admin)]},
),
(
"By township Average hours per day spent on Caring for and helping household children in South Korea ",
{"entities": [(3, 11, admin),(12, 84, theme),(88, 99, region)]},
),
(
"In 2011 Percent change of who believe climate change ",
{"entities": [(3, 7, time),(8, 52, theme)]},
),
(
"Average square footage of houses in South Korea ",
{"entities": [(0, 32, theme),(36, 47, region)]},
),
(
"Average monthly housing cost as percentage of income in 1964 ",
{"entities": [(0, 52, theme),(56, 60, time)]},
),
(
"By province number of fire points ",
{"entities": [(3, 11, admin),(12, 33, theme)]},
),
(
"In 1999 by county Production workers annual wages of Petroleum and coal products manufacturing ",
{"entities": [(3, 7, time),(11, 17, admin),(18, 94, theme)]},
),
(
"In China annual average temperature by county in 1966 ",
{"entities": [(3, 8, region),(9, 35, theme),(39, 45, admin),(49, 53, time)]},
),
(
"In 1960 annual average precipitation by county in USA ",
{"entities": [(3, 7, time),(8, 36, theme),(40, 46, admin),(50, 53, region)]},
),
(
"Energy consumption (per capita) in USA by province ",
{"entities": [(0, 31, theme),(35, 38, region),(42, 50, admin)]},
),
(
"By township in China Average poverty level for household ",
{"entities": [(3, 11, admin),(15, 20, region),(21, 56, theme)]},
),
(
"In 1967 Current operation of governments by township ",
{"entities": [(3, 7, time),(8, 40, theme),(44, 52, admin)]},
),
(
"In U.S. Elementary-secondary revenue from local government ",
{"entities": [(3, 7, region),(8, 58, theme)]},
),
(
"Elementary-secondary revenue from vocational programs in 1995 ",
{"entities": [(0, 53, theme),(57, 61, time)]},
),
(
"In U.S. Average monthly housing cost as percentage of income by township ",
{"entities": [(3, 7, region),(8, 60, theme),(64, 72, admin)]},
),
(
"In South Korea by census tract number of earthquake in 1990 ",
{"entities": [(3, 14, region),(18, 30, admin),(31, 51, theme),(55, 59, time)]},
),
(
"In France annual average precipitation by county ",
{"entities": [(3, 9, region),(10, 38, theme),(42, 48, admin)]},
),
(
"By state in 1999 Interest on general debt of governments in UK ",
{"entities": [(3, 8, admin),(12, 16, time),(17, 56, theme),(60, 62, region)]},
),
(
"Annual average temperature by province ",
{"entities": [(0, 26, theme),(30, 38, admin)]},
),
(
"Capital outlay of elementary-secondary expenditure in 1984 ",
{"entities": [(0, 50, theme),(54, 58, time)]},
),
(
"In U.S. by province federal government expenditure (per capita) in 2006 ",
{"entities": [(3, 7, region),(11, 19, admin),(20, 63, theme),(67, 71, time)]},
),
(
"Household income in 1956 in UK ",
{"entities": [(0, 16, theme),(20, 24, time),(28, 30, region)]},
),
(
"Income Taxes of governments in 1983 in France ",
{"entities": [(0, 27, theme),(31, 35, time),(39, 45, region)]},
),
(
"In South Korea by township in 1993 License Taxes of governments ",
{"entities": [(3, 14, region),(18, 26, admin),(30, 34, time),(35, 63, theme)]},
),
(
"In 1998 annual average temperature ",
{"entities": [(3, 7, time),(8, 34, theme)]},
),
(
"In the United States in 2007 Number of employees of Textile furnishings mills by county ",
{"entities": [(3, 20, region),(24, 28, time),(29, 77, theme),(81, 87, admin)]},
),
(
"In China in 1952 by county annual average temperature ",
{"entities": [(3, 8, region),(12, 16, time),(20, 26, admin),(27, 53, theme)]},
),
(
"Food insecurity rate in 1973 ",
{"entities": [(0, 20, theme),(24, 28, time)]},
),
(
"Number of fire points by state in US in 1955 ",
{"entities": [(0, 21, theme),(25, 30, admin),(34, 36, region),(40, 44, time)]},
),
(
"Total capital expenditures of Other chemical product and preparation manufacturing in 1998 ",
{"entities": [(0, 82, theme),(86, 90, time)]},
),
(
"In South Korea difference in number of people of widowed by state ",
{"entities": [(3, 14, region),(15, 56, theme),(60, 65, admin)]},
),
(
"Total capital expenditures of Motor vehicle parts manufacturing by province in 1971 in USA ",
{"entities": [(0, 63, theme),(67, 75, admin),(79, 83, time),(87, 90, region)]},
),
(
"In South Korea by township Capital outlay of elementary-secondary expenditure in 1968 ",
{"entities": [(3, 14, region),(18, 26, admin),(27, 77, theme),(81, 85, time)]},
),
(
"Exports value of firms by census tract in 1975 ",
{"entities": [(0, 22, theme),(26, 38, admin),(42, 46, time)]},
),
(
"In 1983 difference in population density of White in the United States by province ",
{"entities": [(3, 7, time),(8, 49, theme),(53, 70, region),(74, 82, admin)]},
),
(
"Corporate income tax of governments in 1959 ",
{"entities": [(0, 35, theme),(39, 43, time)]},
),
(
"By township percent of houses with annual income of $300,000 and over in France in 2018 ",
{"entities": [(3, 11, admin),(12, 69, theme),(73, 79, region),(83, 87, time)]},
),
(
"In 2012 by township in China Utility expenditure of governments ",
{"entities": [(3, 7, time),(11, 19, admin),(23, 28, region),(29, 63, theme)]},
),
(
"Average percent of time engaged in Working and work-related activities by state ",
{"entities": [(0, 70, theme),(74, 79, admin)]},
),
(
"By county in 1951 Annual payroll of Animal slaughtering and processing in South Korea ",
{"entities": [(3, 9, admin),(13, 17, time),(18, 70, theme),(74, 85, region)]},
),
(
"Average percent of time engaged in Organizational, civic, and religious activities in 1987 ",
{"entities": [(0, 82, theme),(86, 90, time)]},
),
(
"By province Households with householder living alone in USA in 1998 ",
{"entities": [(3, 11, admin),(12, 52, theme),(56, 59, region),(63, 67, time)]},
),
(
"Estimated annual sales for Health & personal care stores by province in China ",
{"entities": [(0, 56, theme),(60, 68, admin),(72, 77, region)]},
),
(
"Average square footage of houses in UK in 2019 ",
{"entities": [(0, 32, theme),(36, 38, region),(42, 46, time)]},
),
(
"Gross domestic income (nominal or ppp) per capita in US in 1982 by state ",
{"entities": [(0, 49, theme),(53, 55, region),(59, 63, time),(67, 72, admin)]},
),
(
"In South Korea in 1983 by census tract Liquor stores expenditure of governments ",
{"entities": [(3, 14, region),(18, 22, time),(26, 38, admin),(39, 79, theme)]},
),
(
"In USA Households with one or more people 65 years and over ",
{"entities": [(3, 6, region),(7, 59, theme)]},
),
(
"In 2004 GDP (nominal or ppp) ",
{"entities": [(3, 7, time),(8, 28, theme)]},
),
(
"In 2017 in U.S. by county percent of farmland ",
{"entities": [(3, 7, time),(11, 15, region),(19, 25, admin),(26, 45, theme)]},
),
(
"In 2009 in China by township difference in number of people of Under age 18 ",
{"entities": [(3, 7, time),(11, 16, region),(20, 28, admin),(29, 75, theme)]},
),
(
"In 1978 by county Estimated annual sales for Food & beverage stores ",
{"entities": [(3, 7, time),(11, 17, admin),(18, 67, theme)]},
),
(
"Elementary-secondary revenue from vocational programs in 1978 ",
{"entities": [(0, 53, theme),(57, 61, time)]},
),
(
"Annual payroll in 2006 ",
{"entities": [(0, 14, theme),(18, 22, time)]},
),
(
"Average monthly housing cost as percentage of income in the United States in 2011 by township ",
{"entities": [(0, 52, theme),(56, 73, region),(77, 81, time),(85, 93, admin)]},
),
(
"In the United States Households with householder living alone by province ",
{"entities": [(3, 20, region),(21, 61, theme),(65, 73, admin)]},
),
(
"Production workers average for year of Other general purpose machinery manufacturing in 2014 in U.S. by state ",
{"entities": [(0, 84, theme),(88, 92, time),(96, 100, region),(104, 109, admin)]},
),
(
"Number of McDonald's in the United States in 1971 ",
{"entities": [(0, 20, theme),(24, 41, region),(45, 49, time)]},
),
(
"By state in 2001 Population density of Muslim in France ",
{"entities": [(3, 8, admin),(12, 16, time),(17, 45, theme),(49, 55, region)]},
),
(
"In 1976 by census tract availability of safe drinking water in France ",
{"entities": [(3, 7, time),(11, 23, admin),(24, 59, theme),(63, 69, region)]},
),
(
"In the United States Number of paid employees by census tract ",
{"entities": [(3, 20, region),(21, 45, theme),(49, 61, admin)]},
),
(
"Average percent of time engaged in by womenCaring for and helping household children in 1959 ",
{"entities": [(0, 84, theme),(88, 92, time)]},
),
(
"In USA by province Household income ",
{"entities": [(3, 6, region),(10, 18, admin),(19, 35, theme)]},
),
(
"Annual average precipitation by township in 2006 in France ",
{"entities": [(0, 28, theme),(32, 40, admin),(44, 48, time),(52, 58, region)]},
),
(
"Percent of farms with female principal operator in France in 1963 by province ",
{"entities": [(0, 47, theme),(51, 57, region),(61, 65, time),(69, 77, admin)]},
),
(
"Population density of Jewish in 1981 in US by township ",
{"entities": [(0, 28, theme),(32, 36, time),(40, 42, region),(46, 54, admin)]},
),
(
"Average family size by census tract in US in 1959 ",
{"entities": [(0, 19, theme),(23, 35, admin),(39, 41, region),(45, 49, time)]},
),
(
"Percent change of people enrolled in Nursery school, people enrolled in preschool in South Korea by state ",
{"entities": [(0, 81, theme),(85, 96, region),(100, 105, admin)]},
),
(
"In the United States in 2016 gun violence rate ",
{"entities": [(3, 20, region),(24, 28, time),(29, 46, theme)]},
),
(
"Percent of houses with annual income of $50,000 and less in USA by state in 2003 ",
{"entities": [(0, 56, theme),(60, 63, region),(67, 72, admin),(76, 80, time)]},
),
(
"By township in 2006 Elementary-secondary revenue from other state aid ",
{"entities": [(3, 11, admin),(15, 19, time),(20, 69, theme)]},
),
(
"In France number of fire points ",
{"entities": [(3, 9, region),(10, 31, theme)]},
),
(
"In 2009 by county number of fire points in South Korea ",
{"entities": [(3, 7, time),(11, 17, admin),(18, 39, theme),(43, 54, region)]},
),
(
"In 1999 Exports value of firms ",
{"entities": [(3, 7, time),(8, 30, theme)]},
),
(
"Elementary-secondary revenue from property taxes in 1982 ",
{"entities": [(0, 48, theme),(52, 56, time)]},
),
(
"In 2005 by province Elementary-secondary revenue from parent government contributions in the United States ",
{"entities": [(3, 7, time),(11, 19, admin),(20, 85, theme),(89, 106, region)]},
),
(
"In 2017 by township in South Korea Age of householder ",
{"entities": [(3, 7, time),(11, 19, admin),(23, 34, region),(35, 53, theme)]},
),
(
"Estimated annual sales for Elect. shopping & m/o houses in 1986 by township in the United States ",
{"entities": [(0, 55, theme),(59, 63, time),(67, 75, admin),(79, 96, region)]},
),
(
"Insurance trust expenditure of governments in China ",
{"entities": [(0, 42, theme),(46, 51, region)]},
),
(
"Capital outlay of elementary-secondary expenditure in 1973 in US by province ",
{"entities": [(0, 50, theme),(54, 58, time),(62, 64, region),(68, 76, admin)]},
),
(
"Households with male householder, no wife present, family in U.S. ",
{"entities": [(0, 57, theme),(61, 65, region)]},
),
(
"By county percent of houses with annual income of $300,000 and over ",
{"entities": [(3, 9, admin),(10, 67, theme)]},
),
(
"In USA in 1959 number of fixed residential broadband providers ",
{"entities": [(3, 6, region),(10, 14, time),(15, 62, theme)]},
),
(
"Median household income in 1995 ",
{"entities": [(0, 23, theme),(27, 31, time)]},
),
(
"By province Annual payroll of Clay product and refractory manufacturing ",
{"entities": [(3, 11, admin),(12, 71, theme)]},
),
(
"Elementary-secondary revenue from state sources in South Korea ",
{"entities": [(0, 47, theme),(51, 62, region)]},
),
(
"By township in South Korea in 1987 number of fire points ",
{"entities": [(3, 11, admin),(15, 26, region),(30, 34, time),(35, 56, theme)]},
),
(
"By province helicobacter pylori rate ",
{"entities": [(3, 11, admin),(12, 36, theme)]},
),
(
"Number of paid employees by census tract ",
{"entities": [(0, 24, theme),(28, 40, admin)]},
),
(
"License Taxes of governments in China ",
{"entities": [(0, 28, theme),(32, 37, region)]},
),
(
"In 1999 by province Elementary-secondary revenue from school lunch charges ",
{"entities": [(3, 7, time),(11, 19, admin),(20, 74, theme)]},
),
(
"In US number of McDonald's by county in 1966 ",
{"entities": [(3, 5, region),(6, 26, theme),(30, 36, admin),(40, 44, time)]},
),
(
"In U.S. by province in 2000 Total cost of materials of Cement and concrete product manufacturing ",
{"entities": [(3, 7, region),(11, 19, admin),(23, 27, time),(28, 96, theme)]},
),
(
"Estimated annual sales for Clothing & clothing accessories stores in 1960 by state ",
{"entities": [(0, 65, theme),(69, 73, time),(77, 82, admin)]},
),
(
"By county in 2000 GDP (nominal or ppp) per capita in China ",
{"entities": [(3, 9, admin),(13, 17, time),(18, 49, theme),(53, 58, region)]},
),
(
"By state in 1958 in UK Sales, receipts, or value of shipments of firms ",
{"entities": [(3, 8, admin),(12, 16, time),(20, 22, region),(23, 70, theme)]},
),
(
"Price of land in 1993 in the United States ",
{"entities": [(0, 13, theme),(17, 21, time),(25, 42, region)]},
),
(
"In 1983 Elementary-secondary revenue from compensatory programs by census tract ",
{"entities": [(3, 7, time),(8, 63, theme),(67, 79, admin)]},
),
(
"Percent change of Under age 18 by township in US in 2007 ",
{"entities": [(0, 30, theme),(34, 42, admin),(46, 48, region),(52, 56, time)]},
),
(
"In 1996 in South Korea Estimated annual sales for Beer, wine & liquor stores ",
{"entities": [(3, 7, time),(11, 22, region),(23, 76, theme)]},
),
(
"In 1989 in UK Production workers average for year of Machine shops; turned product; and screw, nut, and bolt manscrewufacturing ",
{"entities": [(3, 7, time),(11, 13, region),(14, 127, theme)]},
),
(
"Average percent of time engaged in by womenCaring for and helping nonhousehold children in 2005 by county ",
{"entities": [(0, 87, theme),(91, 95, time),(99, 105, admin)]},
),
(
"By census tract annual average precipitation in 1985 ",
{"entities": [(3, 15, admin),(16, 44, theme),(48, 52, time)]},
),
(
"In 1962 by state Number of employees of Electrical equipment manufacturing ",
{"entities": [(3, 7, time),(11, 16, admin),(17, 74, theme)]},
),
(
"In South Korea in 1953 Average hours per day spent on Government services ",
{"entities": [(3, 14, region),(18, 22, time),(23, 73, theme)]},
),
(
"In U.S. by state in 2011 difference in population density of dentist ",
{"entities": [(3, 7, region),(11, 16, admin),(20, 24, time),(25, 68, theme)]},
),
(
"Percent of households above $200k in 2016 ",
{"entities": [(0, 33, theme),(37, 41, time)]},
),
(
"In 1955 in U.S. annual average precipitation by census tract ",
{"entities": [(3, 7, time),(11, 15, region),(16, 44, theme),(48, 60, admin)]},
),
(
"In U.S. by county happiness score in 1999 ",
{"entities": [(3, 7, region),(11, 17, admin),(18, 33, theme),(37, 41, time)]},
),
(
"In 1996 Estimated annual sales for Food & beverage stores in USA by census tract ",
{"entities": [(3, 7, time),(8, 57, theme),(61, 64, region),(68, 80, admin)]},
),
(
"In France Exports value of firms ",
{"entities": [(3, 9, region),(10, 32, theme)]},
),
(
"In the United States annual average temperature by state in 1988 ",
{"entities": [(3, 20, region),(21, 47, theme),(51, 56, admin),(60, 64, time)]},
),
(
"Number of firms in US in 1965 ",
{"entities": [(0, 15, theme),(19, 21, region),(25, 29, time)]},
),
(
"In 1990 number of people of Christian ",
{"entities": [(3, 7, time),(8, 37, theme)]},
),
(
"By province Agriculture exports ",
{"entities": [(3, 11, admin),(12, 31, theme)]},
),
(
"Annual average temperature in 2020 ",
{"entities": [(0, 26, theme),(30, 34, time)]},
),
(
"By province Elementary-secondary revenue from special education in 1998 in the United States ",
{"entities": [(3, 11, admin),(12, 63, theme),(67, 71, time),(75, 92, region)]},
),
(
"In US annual average temperature by state in 1980 ",
{"entities": [(3, 5, region),(6, 32, theme),(36, 41, admin),(45, 49, time)]},
),
(
"In France in 1989 unemployment rate by census tract ",
{"entities": [(3, 9, region),(13, 17, time),(18, 35, theme),(39, 51, admin)]},
),
(
"In UK by county Taxes of governments ",
{"entities": [(3, 5, region),(9, 15, admin),(16, 36, theme)]},
),
(
"In 1966 in South Korea Average percent of time engaged in by womenInterior cleaning by state ",
{"entities": [(3, 7, time),(11, 22, region),(23, 83, theme),(87, 92, admin)]},
),
(
"Production workers annual wages of Aerospace product and parts manufacturing by state ",
{"entities": [(0, 76, theme),(80, 85, admin)]},
),
(
"Life expectancy in 1991 ",
{"entities": [(0, 15, theme),(19, 23, time)]},
),
(
"By county in 1951 Number of firms ",
{"entities": [(3, 9, admin),(13, 17, time),(18, 33, theme)]},
),
(
"By county in 1964 in China Total Taxes of governments ",
{"entities": [(3, 9, admin),(13, 17, time),(21, 26, region),(27, 53, theme)]},
),
(
"Average percent of time engaged in Household management by province in France ",
{"entities": [(0, 55, theme),(59, 67, admin),(71, 77, region)]},
),
(
"By township in 2003 in USA number of earthquake ",
{"entities": [(3, 11, admin),(15, 19, time),(23, 26, region),(27, 47, theme)]},
),
(
"In 2006 by census tract Nonfamily households in South Korea ",
{"entities": [(3, 7, time),(11, 23, admin),(24, 44, theme),(48, 59, region)]},
),
(
"By province in 1985 Elementary-secondary revenue from general formula assistance ",
{"entities": [(3, 11, admin),(15, 19, time),(20, 80, theme)]},
),
(
"Production workers annual hours of Metalworking machinery manufacturing by census tract in USA ",
{"entities": [(0, 71, theme),(75, 87, admin),(91, 94, region)]},
),
(
"Annual average temperature in 1971 ",
{"entities": [(0, 26, theme),(30, 34, time)]},
),
(
"In U.S. annual average temperature ",
{"entities": [(3, 7, region),(8, 34, theme)]},
),
(
"By province number of fixed residential broadband providers in 1976 ",
{"entities": [(3, 11, admin),(12, 59, theme),(63, 67, time)]},
),
(
"Annual payroll in France by province ",
{"entities": [(0, 14, theme),(18, 24, region),(28, 36, admin)]},
),
(
"By census tract federal government expenditure (per capita) ",
{"entities": [(3, 15, admin),(16, 59, theme)]},
),
(
"Individual income tax of governments by township ",
{"entities": [(0, 36, theme),(40, 48, admin)]},
),
(
"In 2018 Households with female householder, no husband present, family in France ",
{"entities": [(3, 7, time),(8, 70, theme),(74, 80, region)]},
),
(
"In U.S. in 1970 Liquor stores expenditure of governments ",
{"entities": [(3, 7, region),(11, 15, time),(16, 56, theme)]},
),
(
"In 2007 in UK Average hours per day spent on Storing interior household items, including food ",
{"entities": [(3, 7, time),(11, 13, region),(14, 93, theme)]},
),
(
"In 1983 in France Exports value of firms by census tract ",
{"entities": [(3, 7, time),(11, 17, region),(18, 40, theme),(44, 56, admin)]},
),
(
"Liquor stores revenue of governments in 1995 by state ",
{"entities": [(0, 36, theme),(40, 44, time),(48, 53, admin)]},
),
(
"Annual average precipitation by township in 1956 ",
{"entities": [(0, 28, theme),(32, 40, admin),(44, 48, time)]},
),
(
"In the United States by province in 2000 Estimated annual sales for Motor vehicle & parts Dealers ",
{"entities": [(3, 20, region),(24, 32, admin),(36, 40, time),(41, 97, theme)]},
),
(
"By township Sales, receipts, or value of shipments of firms ",
{"entities": [(3, 11, admin),(12, 59, theme)]},
),
(
"In the United States by census tract Average hours per day spent on Household and personal e-mail and messages ",
{"entities": [(3, 20, region),(24, 36, admin),(37, 110, theme)]},
),
(
"In 2000 Estimated annual sales for Elect. shopping & m/o houses by province in the United States ",
{"entities": [(3, 7, time),(8, 63, theme),(67, 75, admin),(79, 96, region)]},
),
(
"In South Korea gross domestic income (nominal or ppp) ",
{"entities": [(3, 14, region),(15, 53, theme)]},
),
(
"In South Korea in 1974 by state number of earthquake ",
{"entities": [(3, 14, region),(18, 22, time),(26, 31, admin),(32, 52, theme)]},
),
(
"By county in US Liquor stores revenue of governments ",
{"entities": [(3, 9, admin),(13, 15, region),(16, 52, theme)]},
),
(
"By township in U.S. Household income in 1999 ",
{"entities": [(3, 11, admin),(15, 19, region),(20, 36, theme),(40, 44, time)]},
),
(
"In 2013 in the United States unemployment rate ",
{"entities": [(3, 7, time),(11, 28, region),(29, 46, theme)]},
),
(
"Intergovernmental revenue of governments by census tract ",
{"entities": [(0, 40, theme),(44, 56, admin)]},
),
(
"Average percent of time engaged in by menCaring for and helping household members by census tract in South Korea ",
{"entities": [(0, 81, theme),(85, 97, admin),(101, 112, region)]},
),
(
"Exports value of firms in 1953 ",
{"entities": [(0, 22, theme),(26, 30, time)]},
),
(
"Average square footage of houses in 2015 in the United States ",
{"entities": [(0, 32, theme),(36, 40, time),(44, 61, region)]},
),
(
"Average monthly housing cost in 1953 in China by census tract ",
{"entities": [(0, 28, theme),(32, 36, time),(40, 45, region),(49, 61, admin)]},
),
(
"Production workers annual wages of Other miscellaneous manufacturing in USA ",
{"entities": [(0, 68, theme),(72, 75, region)]},
),
(
"By census tract in 1966 Sales, receipts, or value of shipments of firms in the United States ",
{"entities": [(3, 15, admin),(19, 23, time),(24, 71, theme),(75, 92, region)]},
),
(
"Average hours per day spent on Participating in sports, exercise, and recreation in UK ",
{"entities": [(0, 80, theme),(84, 86, region)]},
),
(
"Production workers annual wages of Computer and electronic product manufacturing in South Korea ",
{"entities": [(0, 80, theme),(84, 95, region)]},
),
(
"Percent change of Hispanic or Latino Origin in 1976 by county in U.S. ",
{"entities": [(0, 43, theme),(47, 51, time),(55, 61, admin),(65, 69, region)]},
),
(
"In 1974 in France Elementary-secondary revenue from vocational programs by county ",
{"entities": [(3, 7, time),(11, 17, region),(18, 71, theme),(75, 81, admin)]},
),
(
"Elementary-secondary revenue from parent government contributions in 1965 by province ",
{"entities": [(0, 65, theme),(69, 73, time),(77, 85, admin)]},
),
(
"In the United States by state in 2016 Number of firms ",
{"entities": [(3, 20, region),(24, 29, admin),(33, 37, time),(38, 53, theme)]},
),
# from here, real titles
(
"States and UTs",
{"entities": [(0,6, admin)]},
),
(
"Example choropleth map",
{"entities": []},
),
(
"lifeExp (2007)",
{"entities": [(0,7,theme),(9,13,time)]},
),
(
"Males per 100 Females",
{"entities": [(0,21,theme)]},
),
(
"Choropleth Map",
{"entities": []},
),
(
"Measles incidence per district",
{"entities": [(0,30,theme)]},
),
(
"Human Development Index, (Statistics reported from UNDP 2001)",
{"entities": [(0,23,theme),(56,60,time)]},
),
(
"Average rate per 10,000 people",
{"entities": [(0,30,theme)]},
),
(
"Dasymetric Map of Annual Avergage DailyTraffic Density",
{"entities": [(18,54,theme)]},
),
(
"Value of Land",
{"entities": [(0,13,theme)]},
),
(
"GIS In Demographics: Visualizing Population Growth & western Migration Over a 200-Year Period",
{"entities": [(33,70,theme)]},
),
(
"CIC Returnable Loans/Borrows by US County",
{"entities": [(0,28,theme),(32,34,region),(35,41,admin)]},
),
(
"Percent of homes $300,000 and over",
{"entities": [(0,34,theme)]},
),
(
"PREDOMIMANT MANUFACTURING ACTIVITY, 1997",
{"entities": [(0,34,theme),(36,40,time)]},
),
(
"Percent of homes less than $50,000",
{"entities": [(0,34,theme)]},
),
(
"New Zealand Suicide Rate by District Health Board (DHB)",
{"entities": [(12,24,theme),(0,11,region)]},
),
(
"World Life Expectancy Map",
{"entities": [(0,21,theme)]},
),
(
"DENSITY OF POPULATION, 2011",
{"entities": [(0,21,theme),(23,27,time)]},
),
(
"POPULATION DENSITY, 2000",
{"entities": [(0,18,theme),(20,24,time)]},
),
(
"Mycobacterium bovis in wildlife",
{"entities": [(0,19,theme)]},
),
(
"Population % white ethnic groups 2001 Southampton output areas",
{"entities": [(0,32,theme),(33,37,time),(38,49,region)]},
),
(
"DG ECHO Dally Map Emergency Response Coordination Centre (ERCC) COVID-19 pandemic worldwide",
{"entities": [(64,81,theme),(82,91,region)]},
),
(
"Population per square mile by state 2000 census figures",
{"entities": [(0,26,theme),(30,35,admin),(36,40,time)]},
),
(
"Populatian Density and Offender Density Rates",
{"entities": [(0,18,theme),(23,45,theme)]},
),
(
"Rhode Island Social Vulnerability",
{"entities": [(0,12,region),(13,33,theme)]},
),
(
"Median Household Income and Unrepaired Sinkholes",
{"entities": [(0,23,theme),(28,48,theme)]},
),
(
"Life Expectancy Indian states 2011-2016, at birth",
{"entities": [(0,15,theme),(16,22,region),(23,29,admin),(30,39,time)]},
),
(
"Population in Europe",
{"entities": [(0,10,theme), (14,20,region)]},
),
(
"Thematic maps choropleth maps",
{"entities": []},
),
(
"District Wise Crime against women in India in 2015",
{"entities": [(0,8,admin),(14,33,theme),(37,42,region),(46,50,time)]},
),
(
"Preventable death rate in London",
{"entities": [(0,22,theme),(26,32,region)]},
),
(
"Share of adults defined as obese, 2016",
{"entities": [(0,32,theme),(34,38,time)]},
),
(
"Reported cases per 100 000 people (as of 04 May 2020)",
{"entities": [(0,33,theme),(41,52,time)]},
),
(
"Human Poverty Index Map",
{"entities": [(0,19,theme)]},
),
(
"Share of adults who are obese, 2016",
{"entities": [(0,29,theme),(31,35,time)]},
),
(
"France's population density, 1821",
{"entities": [(0,27,theme),(29,33,time)]},
),
(
"Percentage of Country's Population Living in Urban Areas, 1960",
{"entities": [(0,56,theme),(58,62,time)]},
),
(
"EU Referendum A Divided Kingdom",
{"entities": []},
),
(
"US Choropleth County Map per State - Unemployment Rates",
{"entities": [(37,55,theme),(0,2,region),(14,20,admin)]},
),
(
"Choropleth Map - Germany by ZIP Codes (\"PLZ 2\")",
{"entities": [(17,24,region)]},
),
(
"Choropleth Map of the United States by congressional districts",
{"entities": [(18,34,region),(39,62,admin)]},
),
(
"Choropleth Map of Brazil",
{"entities": [(18,24,region)]},
),
(
"Percent of Persons Who Are Hispanic or Latino (of any race), Florida by County",
{"entities": [(0,59,theme),(61,68,region),(72,78,region)]},
),
(
"Switzerland's regional demographics Average age in Swiss municipalities, 2015",
{"entities": [(0,11,region),(36,47,theme),(73,77,time),(57,71,admin)]},
),
(
"Homicide rate by municipality, 2008",
{"entities": [(0,8,theme),(17,29,admin),(31,35,time)]},
),
(
"Population",
{"entities": [(0,10,theme)]},
),
(
"2012 Population Estimates",
{"entities": [(0,4,time),(5,25,theme)]},
),
(
"Incidence of total crimes against women in India in 2001",
{"entities": [(0,39,theme),(43,48,region),(52,56,time)]},
),
(
"U.S. Population Change",
{"entities": [(0,4,region),(5,22,theme)]},
),
(
"Population Density (people per square miles)",
{"entities": [(0,44,theme)]},
),
(
"Those who contribute the least greenhouse gases will be most impacted by climate change",
{"entities": []},
),
(
"Figure 1: There is Rapid Population Growth in Areas with Many Vulnerable Species",
{"entities": [(25,42,theme)]},
),
(
"Florida Democratic Primary 2016 Percent of Total Votes by County",
{"entities": [(0,7,region),(27,31,time),(32,54,theme),(58, 64, admin)]},
),
(
"POPULATION DENSITY ASIA 2009",
{"entities": [(0,18,theme),(19,23,region),(24,28,time)]},
),
(
"NYC 311 Street Complaints by Community DIstrict",
{"entities": [(0,3,region),(8,25,theme),(29,47,admin)]},
),
(
"Percent Change, 1990 to 2000 and Population Density, 1990",
{"entities": [(0,14,theme),(16,28,time),(33,51,theme),(53,57,time)]},
),
(
"A: Total number of frauds in 2012 per thousand people by local authority area",
{"entities": [(3,25,theme),(29,33,time),(57,77,admin)]},
),
(
"Tamilnadu District Level Literacy Map",
{"entities": [(0,9,region),(10,18,admin),(25,33,theme)]},
),
(
"OXBRIDGE OFFER RATE PER POPULATION MAP",
{"entities": [(0,34,theme)]},
),
(
"Planted Soybeans | 2008",
{"entities": [(0,16,theme),(19,23,time)]},
),
(
"Deaths among children under 5 due to HIV/AIDS (%)",
{"entities": [(0,49,theme)]},
),
(
"Population undernourished 2004 to 2006 (%)",
{"entities": [(0,25,theme),(26,38,theme)]},
),
(
"Strengths and limitations of the mapping technique",
{"entities": []},
),
(
"Let My People Grow Population Density of African Countries in 2008",
{"entities": [(19,38,theme),(41,58,region),(62,66,time)]},
),
(
"Percent Hispanic or Latino (of any race)",
{"entities": [(0,40,theme)]},
),
(
"Choropleth Map with D3.js and SVG",
{"entities": []},
),
(
"On-Trade (%) by Countries - Alcoholic Drinks",
{"entities": [(28,44,theme),(16,25,admin)]},
),
(
"Choropleth Map - Number of WorldCat Libraries by US County",
{"entities": [(17,45,theme),(49,51,region),(52,58,admin)]},
),
(
"Asian Population in Continental U.S.",
{"entities": [(0,16,theme),(20,36,region)]},
),
(
"Core concerns",
{"entities": []},
),
(
"Example choropleth map",
{"entities": []},
),
(
"Adult obesity by health region 2005",
{"entities": [(0,13,theme),(17,30,admin),(31,35,time)]},
),
(
"Hard to Reach",
{"entities": []},
),
(
"Choropleth Map",
{"entities": []},
),
(
"Measles incidence per district - wave 8",
{"entities": [(0,17,theme),(22,30,admin)]},
),
(
"Human Development Index (Statistics reported from UNDP 2001)",
{"entities": [(0,23,theme),(55,59,time)]},
),
(
"Mortality associated with arterial hypertension Brazilian States, In 2014",
{"entities": [(0,47,theme),(48,64,region),(69,73,time)]},
),
(
"Influenza Patents by Country According to USPTO data",
{"entities": [(0,17,theme),(21,28,admin)]},
),
(
"Value of Land: tax assessment of land, per sqft of dirt",
{"entities": [(0,13,theme)]},
),
(
"Choropleth Map",
{"entities": []},
),
(
"GIS in Demographics: Visualizing Population Growth & Western Migration Over a 200-Year Period",
{"entities": [(33,70,theme)]},
),
(
"Choropleth Map - CIC Returnable Loans/Borrows by US County",
{"entities": [(17,45,theme),(49,51,region),(52,58,admin)]},
),
(
"Percent of homes $300,00 and over",
{"entities": [(0,33,theme)]},
),
(
"POPULATION DENSITY, 2000 Saudi Arabia",
{"entities": [(0,18,theme), (20,24,time), (25,37,region)]},
),
(
"POPULATION DENSITY, 2000 Saudi Arabia",
{"entities": [(0,18,theme), (20,24,time), (25,37,region)]},
),
(
"Population in Europe 1:15 000 000",
{"entities": [(0,10,theme), (14,20,region)]},
),
(
"LIBYA'S POPULATION AND ENERGY PRODUCTION",
{"entities": [(8,40,theme), (0,5,region)]},
),
(
"Thematic maps - choropleth maps",
{"entities": []},
),
(
"MAP ES.1 The Estimated Effects of Water Scarcity on GDP in Year 2050, under Two Policy Regimes",
{"entities": [(13,55,theme), (64,68,time)]},
),
(
"@International Mapping",
{"entities": []},
),
(
"Population in 2008",
{"entities": [(0,10,theme), (14,18,time)]},
),
(
"Map showing the HDI of countries",
{"entities": [(16,19,theme), (23,32,admin)]},
),
(
"Per Capita Income: Per capita income in the past 12 months (in 2011 inflation-adjusted dollars)",
{"entities": [(0,17,theme), (63,67,time)]},
),
(
"Population density of Vancouver (by dissemination areas), 2011",
{"entities": [(0,18,theme), (22,31,region),(36,55,admin),(58,62,time)]},
),
(
"2016 Median income in Pennsylvania counties",
{"entities": [(5,18,theme), (22,34,region),(35,43,admin),(0,4,time)]},
),
(
"Texas Congressional Districts November 2002 Federal Election Results",
{"entities": [(44,68,theme), (0,5,region),(6,29,admin),(30,43,time)]},
),
(
"DISTRIBUTION OF THE SPANISH POPULATION (2005)",
{"entities": [(0,38,theme), (40,44,time)]},
),
(
"Literacy rate map",
{"entities": [(0,13,theme)]},
),
(
"Life Expectancy, Indian states, 2011-2016, at birth",
{"entities": [(0,15,theme), (17,23,region),(24,30,admin),(32,41,time)]},
),
(
"India Population Density Map",
{"entities": [(6,24,theme), (0,5,region)]},
),
(
"Religious Diversity in the U.S., 2010",
{"entities": [(0,19,theme), (23,31,region), (33,37,time)]},
),
(
"Arranged marriages with no consent",
{"entities": [(0,34,theme)]},
),
(
"Average rate per 10,1000 people",
{"entities": [(0,31,theme)]},
),
(
"Dasymetric Map of Annual Avergage DailyTraffic Density",
{"entities": [(18,56,theme)]},
),
(
"Freshwater availability, cubic metres per person and per year, 2007",
{"entities": [(0,23,theme), (63,67,time)]},
),
(
"Mexico City January 2013 to September 2016",
{"entities": [(0,11,region), (12,42,time)]},
),
( # US state
'Share of high school students attending a school with a sworn law enforcement officer',
{"entities": [(0,85,theme)]},
),
(
'Males per 100 Females, Census 2000',
{"entities": [(0,21,theme),(30,34,time)]},
),
(
'US population density',
{"entities": [(0,2,region),(3,21,theme)]},
),
(
'Average Temperature for the US States from July 2015',
{"entities": [(0,19,theme),(24,30,region),(31,37,admin),(43,52,time)]},
),
(
'Population per square mile by state. 2000 census figures.',
{"entities": [(0,26,theme),(30,35,admin),(37,41,time)]},
),
(
'Average annual rainfall across the states of the United States of America',
{"entities": [(0,23,theme),(35,41,admin),(45,73,region)]},
),
(
'Estimated Median Household Income, 2008 Contiguous United States',
{"entities": [(0,33,theme),(35,39,time),(40, 64, region)]},
),
(
'Percent of People Below Poverty Level 2004',
{"entities": [(0,37,theme),(38,42,time)]},
),
(
'Obesity trends * Among U.S. Adults, BRFSS, 2010',
{"entities": [(0,14,theme),(23,27,region),(43,47,time)]},
),
(
'1990 Census Data, % of Population 65 and Older',
{"entities": [(0,4,time),(18,46,theme)]},
),
(
'Choropleth map',
{"entities": []},
),
(
'Choropleth map - CIC returnable loans/borrows by US county',
{"entities": [(17,45,theme),(49,51,region),(52,58,admin)]},
),
(
'Thematic maps - choropleth maps',
{"entities": []},
),
(
'Change in Divorce Rates, Between 1980 and 1990',
{"entities": [(0,23,theme),(25,46,time)]},
),
(
'Figure 1., Percentage of the People Living in Poverty Areas by States: 2006-2010',
{"entities": [(11,59,theme),(63,69,admin),(71,80,time)]},
),
(
'state rankings',
{"entities": [(0,5,admin),(6,14,theme)]},
),
(
'Total withdrawals and deliveries',
{"entities": [(0,32,theme)]},
),
(
'Trump vote',
{"entities": [(0,10,theme)]},
),
(
'Winning margins',
{"entities": [(0,15,theme)]},
),
(
'Trade in goods with China as a % of state GDP',
{"entities": [(0,45,theme)]},
),
(
'U.S. Motor Vehicle Fatalities, 2008',
{"entities": [(0,4,region),(5,29,theme),(31,35,time)]},
),
(
'1170 Coronavirus (COVID-19) Cases in the US',
{"entities": [(5,33,theme),(37,43,region)]},
),
(
'Hazardous Waste Site Installations (1997)',
{"entities": [(0,34,theme),(36,40,time)]},
),
(
'Influenza Research Database Reported Cases 2017-18',
{"entities": [(0,42,theme),(43,50,time)]},
),
(
'COVID-19 in the U.S.',
{"entities": [(0,8,theme),(12,20,region)]},
),
(
'Mexican American Population, 2010 US Census',
{"entities": [(0,27,theme),(29,33,region)]},
),
(
'median income',
{"entities": [(0,13,theme)]},
),
(
'states with most work stress',
{"entities": [(17,28,theme),(0,6,admin)]},
),
(
'Unemployment 2008',
{"entities": [(0,12,theme),(13,17,time)]},
),
(
'Poverty in the United States',
{"entities": [(0,7,theme),(11,28,region)]},
),
(
'Cities supporting emissions reductions (455)',
{"entities": [(0,38,theme)]},
),
(
'Q3 2018 Installed wind power capacity (MW)',
{"entities": [(0,7,time),(8,37,theme)]},
),
(
'Poverty in the United States',
{"entities": [(0,7,theme),(11,28,region)]},
),
(
'2017 Poverty rate in the United States', 
{"entities": [(0,4,time),(5,17,theme),(21,38,region)]},
),
(
'2011 US agriculture exports by state (Hover for breakdown)',
{"entities": [(0,4,time),(5,7,region),(8,27,theme),(31,36,admin)]},
),
(
'Native American Alone/One or More Other Race', 
{"entities": [(0,44,theme)]},
),
(
'States Where Tim Has Spent Time',
{"entities": [(0,31,theme)]},
),
(
'Minority group with highest percent of state population',
{"entities": [(0,55,theme)]},
),
(
'Crime Rates in the US - 2003 vs. Election Results - 2004',
{"entities": [(0,11,theme),(15,21,region),(24,28,time),(33,49,theme),(52,56,time)]},
),
(
'The Wild West Violent Crimes in the Western United States',
{"entities": [(0,28,theme),(32,57,region)]},
),
(
'Median Household Income in the United States: 2015',
{"entities": [(0,23,theme),(27,44,region),(46,50,time)]},
),
(
'NBA players origins per capita',
{"entities": [(0,30,theme)]},
),         
(
'48 states by population',
{"entities": [(13,23,theme),(3,9,admin)]},
),
(
'Word happiness score (the higher the number, the happier)',
{"entities": [(0,57,theme)]},
), 
(
'Number of Persons per Wal-Mart Store', 
{"entities": [(0,36,theme)]},
),
(
'Geo Choropleth Chart: US Venture Capital Landscape 2001',
{"entities": [(25,50,theme),(22,24,region),(51,55,time)]},
),  
(
'U.S. Department of Agriculture - Honey Production, 2009-2013',
{"entities": [(33,49,theme),(9,18,region),(51,60,time)]},
),
(
'Open Source Choropleth Maps',
{"entities": []},
), 
(
'Federal Government Expenditure, Per Capita Ranges by State: Fiscal Year 2009',
{"entities": [(0,42,theme),(53,58,region),(60,76,time)]},
),
(
'Kickstarter USA',
{"entities": [(0,11,theme),(12,15,region)]},
), 
(
'Percent of Popl 65 and Older',
{"entities": [(0,28,theme)]},
), 
(
'U.S. Farmland',
{"entities": [(5,14,theme),(0,4,region)]},
),
(
'Sales by State',
{"entities": [(0,5,theme),(9,14,admin)]},
), 
(
'Well-Being index',
{"entities": [(0,16,theme)]},
),
(
'Food Insecurity Rate',
{"entities": [(0,20,theme)]},
),
(
'Obesity Rate',
{"entities": [(0,12,theme)]},
), 
(
'HSU Alumni Per 100,000 People',
{"entities": [(0,29,theme)]},
),
(
'Figure 2. Percentage of People in Poverty for the United States and Puerto Rico:2013',
{"entities": [(10,41,theme),(46,63,region),(68,79,region),(80,84,time)]},
), 
(
'Regional Heat Map',
{"entities": []},
),
(
'The Number of Multi-racial Housholds per county in the Continental United States', 
{"entities": [(0,36,theme),(41,47,admin),(51,80,region)]},
), 
(
'Percent of 4-year-olds Served by State Pre-K',
{"entities": [(0,29,theme),(33,38,admin)]},
),
(
'One-year forecast change in jobs', 
{"entities": [(0,32,theme)]},
), 
(
'Percentage of People 25 Years and Over Who Have a Bachelor\'s Degree',
{"entities": [(0,68,theme)]},
),
(
'Percentage of 18- to 24-year-olds overweight or obese',
{"entities": [(0,53,theme)]},
), 
(
'Choropleth, 5 Classes, Standard Deviation',
{"entities": []},
),
(
'Estimated % of adults who think global warming is happening, 2014',
{"entities": [(0,59,theme),(61,65,time)]},
), 
(
'Death Rate from Drug Poisoning / Overdose',
{"entities": [(0,41,theme)]},
),
(
'Rate of Temperature Change in the United States, 1901-2015',
{"entities": [(0,26,theme),(30,47,region),(49,58,time)]},
)
]

TEST_DATA= [
    (
    "Population density map, Indian states",
    {"entities": [(0,18,theme), (24,30,region), (31,37,admin)]},
    ),
    (
    "Emergency Response Coordination Centre (ERCC)| DG ECHO Daily Map | 08/08/2020 COVID-19 pandemic worldwide",
    {"entities": [(78,95,theme), (96,105,region), (67,77,time)]},
    ),
    (
    "Maternal Mortality Rate of African Countries",
    {"entities": [(0,23,theme), (27,44,region)]},
    ),
    (
    "Total CO2 emissions (2004)",
    {"entities": [(0,19,theme), (21,25,time)]},
    ),
    (
    "Retailer vs Location Density Hover over a state",
    {"entities": [(0,28,theme)]},
    ),
    (
    "USA by Unemployment %",
    {"entities": [(7,21,theme), (0,3,region)]},
    ),
    (
    "Chennai confirmed COVID-19 cases by zones",
    {"entities": [(8,32,theme), (0,7,region), (36,41,admin)]},
    ),
    (
    "GDP Per Capita Mainland China",
    {"entities": [(0,14,theme), (15,29,region) ]},
    ),
    (
    "Canada Percentage of population aged 14 years and under by 2006 Census Division (CD)",
    {"entities": [(7,55,theme), (0,6,region),(59,63,time) ]},
    ),
    (
    "Number and predominant race of offenders executed in Texas counties in 1982-2013",
    {"entities": [(0,49,theme), (53,58,region),(59,67,admin),(71,80,time) ]},
    ),
    (
    "Number of Mortgages",
    {"entities": [(0,19,theme) ]},
    ),
    (
    "2012 New York City ZCTA Population Estimates",
    {"entities": [(19,44,theme),(0,4,time),(5,18,region) ] },
    ),
    (
    "Life expectancy at birth in Europe (both sexes, 2015)",
    {"entities": [(0,24,theme),(48,52,time),(28,34,region) ] },
    ),
    (
    "Number of schools expressing interest in academy status, by education authority",
    {"entities": [(0,55,theme),(60,79,admin)] },
    ),
    (
    "2010 Japan Population Estimates",
    {"entities": [(11,31,theme),(0,4,time),(5,10,region)] },
    ),
    (
    "Africa Population Density",
    {"entities": [(7,25,theme),(0,6,region)] },
    ),
    (
    "PHOTOVOLTAIC POWER POTENTIAL AFRICA",
    {"entities": [(0,28,theme),(29,35,region)] },
    ),
    (
    "PHOTOVOLTAIC POWER POTENTIAL AFRICA",
    {"entities": [(0,28,theme),(29,35,region)] },
    ),
    (
    "Africa Life Expectancy at Birth",
    {"entities": [(7,31,theme),(0,6,region)] },
    ),
    (
    "ALL INVASIVE CANCERS (C00-C42, C45-C96) 2005-2015",
    {"entities": [(0,20,theme),(40,49,time)] },
    ),
    (
    "POPULATION DENSITY, 2000 Argentina",
    {"entities": [(0,18,theme),(20,24,time),(25,34,region)] },
    ),
    (
    "Black Population in the Continental U.S. in 2000",
    {"entities": [(0,16,theme),(44,48,time),(20,40,region)] },
    ),
    (
    "Reported coronavirus cases worldwide As of March 17, 2020",
    {"entities": [(0,26,theme),(43,57,time),(27,36,region)] },
    ),
    (
    "Coronavirus in China: 24th February 2020",
    {"entities": [(0,11,theme),(22,40,time),(15,20,region)] },
    ),
    (
    "Spending by overseas residents 2003",
    {"entities": [(0,30,theme),(31,35,time)] },
    ),
    (
    "Brazil: Corn (First Season) Production",
    {"entities": [(8,38,theme),(0,6,region)] },
    ),
    (
    "Births per ZIP Code in California (2007)",
    {"entities": [(0,6,theme),(11,19,admin),(23, 33, region),(35,39,time)] },
    ),
    (
    "2012 California County Population Estimates",
    {"entities": [(0,4,time),(5,15,region),(16, 22, admin),(23,33,theme)] },
    ),
    (
    "Fig 1: Rate of unvaccinated population by counties, Texas 2017",
    {"entities": [(58,62,time),(52,57,region),(42, 50, admin),(7,38,theme)] },
    ),
    (
    "Real boundaries",
    {"entities": [] },
    ),
    (
    "% who have already felt negative effects from climate change",
    {"entities": [(0,60,theme)] },
    ),
    (
    "Distribution Map of Hazy Weather in China",
    {"entities": [(20,32,theme),(36,41,region)] },
    ),
    (
    "Population",
    {"entities": [(0,10,theme)] },
    ),
    (
    "Nearest GDP equivalents",
    {"entities": [(0,23,theme)] },
    ),
    (
    "AURIN Australian Urban Research Infrastructure Network Portal",
    {"entities": [] },
    ),
    (
    "All Causes, Under 1 year, Persons Deaths per 1,000 live births, by Statistical Division",
    {"entities": [(0,62,theme),(67,87,admin)] },
    ),
    (
    "Choropleth map of german population",
    {"entities": [(25,35,theme),(18,24,region)] },
    ),
    (
    "Population Density, Michigan, 1999",
    {"entities": [(0,18,theme),(20,28,region),(30,34,time)] },
    ),
    (
    "Santiago, Chile, by Income (in CLP$)",
    {"entities": [(20,26,theme),(0,15,region)]},
    ),
    (
    "Precipitation in Greece",
    {"entities": [(0,13,theme),(17,23,region)]},
    ),
    (
    "Percentage of Country's Population Living in Urban Areas, 2016",
    {"entities": [(0,56,theme),(58,62,time)]},
    ),
    (
    "Preliminary map: all areas",
    {"entities": []},
    ),
    (
    "Where the Coronavirus Has Been Confirmed Locations by number of confirmed COVID-19 cases*",
    {"entities": [(54,88,theme)]},
    ),
    (
    "TOTAL REPORTED CASES OF COVID-19 BY REGIONAL HEALTH AUTHORITY AREAS IN CANADA",
    {"entities": [(0,32,theme),(36,67,admin),(71,77,region)]},
    ),
    (
    "Confirmed Cases of COVID-19 in the United States cases per 100,000 population",
    {"entities": [(0,27,theme),(31,48,region)]},
    ),
    (
    "Figure 1. Distribution of confirmed and probable cases of COVID-19 by province or territory in Canada as of March 25, 2020, 6:00 pm EDT",
    {"entities": [(26,66,theme),(70,91,admin),(95,101,region),(108,135,time)]},
    ),
    (
    "Population Density in New England in 2000 in Persons per Square Mile",
    {"entities": [(0,18,theme),(22,33,region),(37,41,time)]},
    ),
    (
    "2014-2015 Immunization Percentages in California Child Care Facilities",
    {"entities": [(10,34,theme),(38,48,region),(0,9,time)]},
    ),
    (
    "Regional IQ in Russia",
    {"entities": [(0,11,theme),(15,21,region)]},
    ),
    (
    "Sales by Territory (click Territory)",
    {"entities": [(0,5,theme),(9,18,admin)]},)
    ]

@plac.annotations(
    model=(nlpTest, "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model="en_core_web_sm", new_model_name="animal", output_dir=None, n_iter=50):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")

    ner.add_label(theme)  # add new entity label to entity recognizer
    # ner.add_label(region)  # add new entity label to entity recognizer
    # ner.add_label(time)  # add new entity label to entity recognizer
    ner.add_label(admin)  # add new entity label to entity recognizer

    # Adding extraneous labels shouldn't mess anything up
    ner.add_label("VEGETABLE")
    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [
        pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            batches = minibatch(TRAIN_DATA, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer,
                           drop=0.35, losses=losses)
            print("Losses", losses)

    # test the trained model
    test_text_list = ["By state Average percent of time engaged in Vehicles in South Korea in 2004 ",
    "In 1964 Household income ","By census tract in France in 1972 Household income ",
    "By township in 1969 Estimated annual sales for Electronics & appliance stores ",
    "By state Average monthly housing cost as percentage of income ",
    "By township rate of male in Canada ","Number of firms in the United States ",
    "In the UK in 1955 number of cell phones per 100 person ",
    "By county in the UK Average hours per day by women spent on Consumer goods purchases ",
    "Adult obesity by state in 1973 "]
    
    realTitles = [
    "Population density map, Indian states",
    "Emergency Response Coordination Centre (ERCC)| DG ECHO Daily Map | 08/08/2020 COVID-19 pandemic worldwide",
    "Maternal Mortality Rate of African Countries",
    "Total CO2 emissions (2004)",
    "Retailer vs Location Density Hover over a state",
    "USA by Unemployment %",
    "Chennai confirmed COVID-19 cases by zones",
    "GDP Per Capita Mainland China",
    "Canada Percentage of population aged 14 years and under by 2006 Census Division (CD)",
    "Number and predominant race of offenders executed in Texas counties in 1982-2013",
    "Number of Mortgages",
    "2012 New York City ZCTA Population Estimates",
    "Life expectancy at birth in Europe (both sexes, 2015)",
    "Number of schools expressing interest in academy status, by education authority",
    "2010 Japan Population Estimates",
    "Africa Population Density",
    "PHOTOVOLTAIC POWER POTENTIAL AFRICA",
    "Africa Life Expectancy at Birth",
    "ALL INVASIVE CANCERS (C00-C42, C45-C96) 2005-2015",
    "POPULATION DENSITY, 2000 Argentina",
    "Black Population in the Continental U.S. in 2000",
    "Reported coronavirus cases worldwide As of March 17, 2020",
    "Coronavirus in China: 24th February 2020",
    "Spending by overseas residents 2003",
    "Brazil: Corn (First Season) Production",
    "Births per ZIP Code in California (2007)",
    "2012 California County Population Estimates",
    "Fig 1: Rate of unvaccinated population by counties, Texas 2017",
    "Real boundaries",
    "% who have already felt negative effects from climate change",
    "Distribution Map of Hazy Weather in China",
    "Population",
    "Nearest GDP equivalents",
    "AURIN Australian Urban Research Infrastructure Network Portal",
    "All Causes, Under 1 year, Persons Deaths per 1,000 live births, by Statistical Division",
    "Choropleth map of german population",
    "Population Density, Michigan, 1999",
    "Santiago, Chile, by Income (in CLP$)",
    "Precipitation in Greece",
    "Percentage of Country's Population Living in Urban Areas, 2016",
    "Preliminary map: all areas",
    "Where the Coronavirus Has Been Confirmed Locations by number of confirmed COVID-19 cases*",
    "TOTAL REPORTED CASES OF COVID-19 BY REGIONAL HEALTH AUTHORITY AREAS IN CANADA",
    "Confirmed Cases of COVID-19 in the United States cases per 100,000 population",
    "Figure 1. Distribution of confirmed and probable cases of COVID-19 by province or territory in Canada as of March 25, 2020, 6:00 pm EDT",
    "Population Density in New England in 2000 in Persons per Square Mile",
    "2014-2015 Immunization Percentages in California Child Care Facilities",
    "Regional IQ in Russia",
    "Sales by Territory (click Territory)"
    ]
    for test_text in realTitles:
        doc = nlp(test_text)
        print("Entities in '%s'" % test_text)
        for ent in doc.ents:
            print(ent.label_, ent.text)
    

    # save model to output directory
    with open('C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Name Entity Recognition\\en_core_web_sm_THEME.pkl', 'wb') as f:
        pickle.dump(nlp,f)

    results = evaluate(nlp, TEST_DATA)

    with open('C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Name Entity Recognition\\results.pkl', 'wb') as f:
        pickle.dump(results,f)

    output_dir = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Name Entity Recognition'
    new_model_name = 'en_core_web_sm_THEME'
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # Check the classes have loaded back consistently
        assert nlp2.get_pipe("ner").move_names == move_names
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)

if __name__ == "__main__":
    plac.call(main)
