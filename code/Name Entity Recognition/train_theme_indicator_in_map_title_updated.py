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

TRAIN_DATA = [
(
"In 1956 in US by county difference in population density of Christian ",
{"entities": [(3, 7, time),(11, 13, region),(17, 23, admin),(24, 69, theme)]},
),
(
"Water Temperature in USA in 1981 by township ",
{"entities": [(0, 17, theme),(21, 24, region),(28, 32, time),(36, 44, admin)]},
),
(
"In USA by county Mercury Effects in 1966 ",
{"entities": [(3, 6, region),(10, 16, admin),(17, 32, theme),(36, 40, time)]},
),
(
"Crime rate by census tract ",
{"entities": [(0, 10, theme),(14, 26, admin)]},
),
(
"In Canada Core Consumer Prices ",
{"entities": [(3, 9, region),(10, 30, theme)]},
),
(
"By province in South Korea area of Mixed Forest (km2) in 1958 ",
{"entities": [(3, 11, admin),(15, 26, region),(27, 53, theme),(57, 61, time)]},
),
(
"In 1974 in China number of people of people enrolled in Nursery school, people enrolled in preschool by state ",
{"entities": [(3, 7, time),(11, 16, region),(17, 100, theme),(104, 109, admin)]},
),
(
"Households with one or more people 65 years and over by state in UK ",
{"entities": [(0, 52, theme),(56, 61, admin),(65, 67, region)]},
),
(
"Production workers annual hours of Household and institutional furniture and kitchen cabinet manufacturing in the United States ",
{"entities": [(0, 106, theme),(110, 127, region)]},
),
(
"In China difference in number of people of males 15 years and over ",
{"entities": [(3, 8, region),(9, 66, theme)]},
),
(
"Private Sector Credit in France in 1961 by state ",
{"entities": [(0, 21, theme),(25, 31, region),(35, 39, time),(43, 48, admin)]},
),
(
"In 1951 by township Exports value of firms in South Korea ",
{"entities": [(3, 7, time),(11, 19, admin),(20, 42, theme),(46, 57, region)]},
),
(
"In Canada Population density of women that were screened for breast and cervical cancer by jurisdiction in 1966 ",
{"entities": [(3, 9, region),(10, 103, theme),(107, 111, time)]},
),
(
"By township in 1971 Estimated annual sales for Health & personal care stores ",
{"entities": [(3, 11, admin),(15, 19, time),(20, 76, theme)]},
),
(
"By state Average hours per day spent on Medical and care services in 1963 ",
{"entities": [(3, 8, admin),(9, 65, theme),(69, 73, time)]},
),
(
"Average percent of time engaged in by menRelaxing and leisure in USA in 1963 ",
{"entities": [(0, 61, theme),(65, 68, region),(72, 76, time)]},
),
(
"In 1986 by county Total capital expenditures of Steel product manufacturing from purchased steel in the United States ",
{"entities": [(3, 7, time),(11, 17, admin),(18, 96, theme),(100, 117, region)]},
),
(
"By census tract in the United States difference in population density of per Walmart store ",
{"entities": [(3, 15, admin),(19, 36, region),(37, 90, theme)]},
),
(
"By state in 1964 Mold in South Korea ",
{"entities": [(3, 8, admin),(12, 16, time),(17, 21, theme),(25, 36, region)]},
),
(
"Annual payroll in U.S. ",
{"entities": [(0, 14, theme),(18, 22, region)]},
),
(
"In 2000 Manufacturing Corporations Quarterly After-Tax Profits by township in US ",
{"entities": [(3, 7, time),(8, 62, theme),(66, 74, admin),(78, 80, region)]},
),
(
"In the United States Estimated annual sales for Food & beverage stores ",
{"entities": [(3, 20, region),(21, 70, theme)]},
),
(
"By province in US Dioxin Effects in 2003 ",
{"entities": [(3, 11, admin),(15, 17, region),(18, 32, theme),(36, 40, time)]},
),
(
"In 1962 Crude Oil Production by township in China ",
{"entities": [(3, 7, time),(8, 28, theme),(32, 40, admin),(44, 49, region)]},
),
(
"By census tract in 1963 Household income in U.S. ",
{"entities": [(3, 15, admin),(19, 23, time),(24, 40, theme),(44, 48, region)]},
),
(
"In Canada Mercury in 2000 by state ",
{"entities": [(3, 9, region),(10, 17, theme),(21, 25, time),(29, 34, admin)]},
),
(
"Difference in population density of People who is infected by HIV in 1998 ",
{"entities": [(0, 65, theme),(69, 73, time)]},
),
(
"Average poverty level for household in the United States ",
{"entities": [(0, 35, theme),(39, 56, region)]},
),
(
"By county Production workers annual wages of Household appliance manufacturing ",
{"entities": [(3, 9, admin),(10, 78, theme)]},
),
(
"External Debt in 2000 ",
{"entities": [(0, 13, theme),(17, 21, time)]},
),
(
"In 1968 Production workers average for year of Clay product and refractory manufacturing in South Korea by census tract ",
{"entities": [(3, 7, time),(8, 88, theme),(92, 103, region),(107, 119, admin)]},
),
(
"In US Average percent of time engaged in by menCaring for and helping nonhousehold adults in 1958 ",
{"entities": [(3, 5, region),(6, 89, theme),(93, 97, time)]},
),
(
"In 1995 Average percent of time engaged in by menCaring for and helping household children in USA by census tract ",
{"entities": [(3, 7, time),(8, 90, theme),(94, 97, region),(101, 113, admin)]},
),
(
"By province Government Budget in US ",
{"entities": [(3, 11, admin),(12, 29, theme),(33, 35, region)]},
),
(
"Average hours per day by women spent on Interior maintenance, repair, and decoration in France ",
{"entities": [(0, 84, theme),(88, 94, region)]},
),
(
"In China Estimated annual sales for Furniture & home furn. Stores by state ",
{"entities": [(3, 8, region),(9, 65, theme),(69, 74, admin)]},
),
(
"Average age by province ",
{"entities": [(0, 11, theme),(15, 23, admin)]},
),
(
"Elementary-secondary revenue from general formula assistance by province ",
{"entities": [(0, 60, theme),(64, 72, admin)]},
),
(
"In the United States by county Assistance and subsidies of governments ",
{"entities": [(3, 20, region),(24, 30, admin),(31, 70, theme)]},
),
(
"In 1999 in Canada Exports value of firms ",
{"entities": [(3, 7, time),(11, 17, region),(18, 40, theme)]},
),
(
"Total cost of materials of Leather and hide tanning and finishing in 2007 ",
{"entities": [(0, 65, theme),(69, 73, time)]},
),
(
"Sales, receipts, or value of shipments of firms in South Korea ",
{"entities": [(0, 47, theme),(51, 62, region)]},
),
(
"By township Average hours per day by women spent on Care for animals and pets, not veterinary care ",
{"entities": [(3, 11, admin),(12, 98, theme)]},
),
(
"Annual payroll by state ",
{"entities": [(0, 14, theme),(18, 23, admin)]},
),
(
"In the United States Pharmaceutical hazardous wastes by province ",
{"entities": [(3, 20, region),(21, 52, theme),(56, 64, admin)]},
),
(
"By county Lead ",
{"entities": [(3, 9, admin),(10, 14, theme)]},
),
(
"Difference in number of people of People who is infected by HIV in 1986 ",
{"entities": [(0, 63, theme),(67, 71, time)]},
),
(
"Percent change of widowed in 1981 ",
{"entities": [(0, 25, theme),(29, 33, time)]},
),
(
"Area of Estuarine Scrub/Shrub (km2) by state in US ",
{"entities": [(0, 35, theme),(39, 44, admin),(48, 50, region)]},
),
(
"Average hours per day by women spent on Travel related to caring for and helping household members in 1977 by census tract in USA ",
{"entities": [(0, 98, theme),(102, 106, time),(110, 122, admin),(126, 129, region)]},
),
(
"Pesticide Chemicals in 2016 in the United States ",
{"entities": [(0, 19, theme),(23, 27, time),(31, 48, region)]},
),
(
"In 2015 in U.S. by county number of multi-racial households ",
{"entities": [(3, 7, time),(11, 15, region),(19, 25, admin),(26, 59, theme)]},
),
(
"In U.S. by census tract Average percent of time engaged in by womenPlaying with household children, not sports ",
{"entities": [(3, 7, region),(11, 23, admin),(24, 110, theme)]},
),
(
"Average percent of time engaged in by menCaring for and helping nonhousehold members in 1983 in Canada ",
{"entities": [(0, 84, theme),(88, 92, time),(96, 102, region)]},
),
(
"Number of paid employees in 1959 by county ",
{"entities": [(0, 24, theme),(28, 32, time),(36, 42, admin)]},
),
(
"Estimated annual sales for Other general merch. Stores by province in USA ",
{"entities": [(0, 54, theme),(58, 66, admin),(70, 73, region)]},
),
(
"In 2013 by state Percent of population of people who are confirm to be infected by 2019-Nov Coronavirus ",
{"entities": [(3, 7, time),(11, 16, admin),(17, 103, theme)]},
),
(
"Number of employees of Other general purpose machinery manufacturing in U.S. in 1976 ",
{"entities": [(0, 68, theme),(72, 76, region),(80, 84, time)]},
),
(
"In China life expectancy by state ",
{"entities": [(3, 8, region),(9, 24, theme),(28, 33, admin)]},
),
(
"In UK by township Average percent of time engaged in by menRelaxing and leisure ",
{"entities": [(3, 5, region),(9, 17, admin),(18, 79, theme)]},
),
(
"By township Percent change of people below poverty level in 1979 ",
{"entities": [(3, 11, admin),(12, 56, theme),(60, 64, time)]},
),
(
"In 1979 Elementary-secondary revenue from special education in China ",
{"entities": [(3, 7, time),(8, 59, theme),(63, 68, region)]},
),
(
"In 1958 Consumer Spending by county ",
{"entities": [(3, 7, time),(8, 25, theme),(29, 35, admin)]},
),
(
"By census tract number of people of separated in 2018 ",
{"entities": [(3, 15, admin),(16, 45, theme),(49, 53, time)]},
),
(
"In 1999 Average percent of time engaged in Volunteering (organizational and civic activities) by province in China ",
{"entities": [(3, 7, time),(8, 93, theme),(97, 105, admin),(109, 114, region)]},
),
(
"Estimated annual sales for Nonstore retailers in 2006 in U.S. ",
{"entities": [(0, 45, theme),(49, 53, time),(57, 61, region)]},
),
(
"In 1997 area of Woody Wetlands (km2) in China by province ",
{"entities": [(3, 7, time),(8, 36, theme),(40, 45, region),(49, 57, admin)]},
),
(
"In France Average hours per day by women spent on Volunteering (organizational and civic activities) in 2009 by county ",
{"entities": [(3, 9, region),(10, 100, theme),(104, 108, time),(112, 118, admin)]},
),
(
"In Canada Production workers average for year of Other miscellaneous manufacturing in 2012 by county ",
{"entities": [(3, 9, region),(10, 82, theme),(86, 90, time),(94, 100, admin)]},
),
(
"Other taxes of governments in South Korea ",
{"entities": [(0, 26, theme),(30, 41, region)]},
),
(
"By state in USA Exports value of firms in 2017 ",
{"entities": [(3, 8, admin),(12, 15, region),(16, 38, theme),(42, 46, time)]},
),
(
"In 1957 Number of paid employees in US ",
{"entities": [(3, 7, time),(8, 32, theme),(36, 38, region)]},
),
(
"In 1981 Sales, receipts, or value of shipments of firms in South Korea by county ",
{"entities": [(3, 7, time),(8, 55, theme),(59, 70, region),(74, 80, admin)]},
),
(
"In 2003 by state Annual payroll in the United States ",
{"entities": [(3, 7, time),(11, 16, admin),(17, 31, theme),(35, 52, region)]},
),
(
"In France Intergovernmental revenue of governments in 1986 by census tract ",
{"entities": [(3, 9, region),(10, 50, theme),(54, 58, time),(62, 74, admin)]},
),
(
"By county Wood Burning Stoves and Appliances ",
{"entities": [(3, 9, admin),(10, 44, theme)]},
),
(
"Average percent of time engaged in by menEating and drinking by province ",
{"entities": [(0, 60, theme),(64, 72, admin)]},
),
(
"Average year built in South Korea ",
{"entities": [(0, 18, theme),(22, 33, region)]},
),
(
"Number of people of Native Hawaiian and Other Pacific Islander in China by township in 1963 ",
{"entities": [(0, 62, theme),(66, 71, region),(75, 83, admin),(87, 91, time)]},
),
(
"By province Average percent of time engaged in by menFood and drink preparation in 1992 in Canada ",
{"entities": [(3, 11, admin),(12, 79, theme),(83, 87, time),(91, 97, region)]},
),
(
"By county Gross National Product in 1956 ",
{"entities": [(3, 9, admin),(10, 32, theme),(36, 40, time)]},
),
(
"By township in 1979 in UK Average hours per day by men spent on Participating in performance and cultural ",
{"entities": [(3, 11, admin),(15, 19, time),(23, 25, region),(26, 105, theme)]},
),
(
"In 1975 in U.S. Average hours per day by women spent on Helping household adults by township ",
{"entities": [(3, 7, time),(11, 15, region),(16, 80, theme),(84, 92, admin)]},
),
(
"By state in 2002 Average percent of time engaged in Caring for and helping nonhousehold children ",
{"entities": [(3, 8, admin),(12, 16, time),(17, 96, theme)]},
),
(
"By township Interest Rate ",
{"entities": [(3, 11, admin),(12, 25, theme)]},
),
(
"Difference in number of people of Native Hawaiian and Other Pacific Islander in 2016 in U.S. ",
{"entities": [(0, 76, theme),(80, 84, time),(88, 92, region)]},
),
(
"Race diversity index in 1978 ",
{"entities": [(0, 20, theme),(24, 28, time)]},
),
(
"Estimated annual sales for Other general merch. Stores by census tract ",
{"entities": [(0, 54, theme),(58, 70, admin)]},
),
(
"Number of paid employees by county ",
{"entities": [(0, 24, theme),(28, 34, admin)]},
),
(
"In U.S. Annual payroll ",
{"entities": [(3, 7, region),(8, 22, theme)]},
),
(
"Annual payroll in France in 2004 ",
{"entities": [(0, 14, theme),(18, 24, region),(28, 32, time)]},
),
(
"In UK in 1981 Total Housing Inventory by township ",
{"entities": [(3, 5, region),(9, 13, time),(14, 37, theme),(41, 49, admin)]},
),
(
"By county Miscellaneous general revenue of governments ",
{"entities": [(3, 9, admin),(10, 54, theme)]},
),
(
"By census tract in 2005 percent of farmland ",
{"entities": [(3, 15, admin),(19, 23, time),(24, 43, theme)]},
),
(
"By state in U.S. Total cost of materials of Fabric mills in 1980 ",
{"entities": [(3, 8, admin),(12, 16, region),(17, 56, theme),(60, 64, time)]},
),
(
"By township Terms of Trade in the United States ",
{"entities": [(3, 11, admin),(12, 26, theme),(30, 47, region)]},
),
(
"By township in 2004 in UK Average hours per day by women spent on Attending class ",
{"entities": [(3, 11, admin),(15, 19, time),(23, 25, region),(26, 81, theme)]},
),
(
"Estimated annual sales for Elect. shopping & m/o houses in 1960 by county in the United States ",
{"entities": [(0, 55, theme),(59, 63, time),(67, 73, admin),(77, 94, region)]},
),
(
"In China by province License taxes of governments in 2006 ",
{"entities": [(3, 8, region),(12, 20, admin),(21, 49, theme),(53, 57, time)]},
),
(
"In USA by province in 1959 Population density of now married, except separated ",
{"entities": [(3, 6, region),(10, 18, admin),(22, 26, time),(27, 78, theme)]},
),
(
"Social Security Rate For Companies in 1996 ",
{"entities": [(0, 34, theme),(38, 42, time)]},
),
(
"Average percent of time engaged in Travel related to work in 2014 by province in Canada ",
{"entities": [(0, 57, theme),(61, 65, time),(69, 77, admin),(81, 87, region)]},
),
(
"By state General expenditure of governments ",
{"entities": [(3, 8, admin),(9, 43, theme)]},
),
(
"Estimated annual sales for New car dealers in 1967 ",
{"entities": [(0, 42, theme),(46, 50, time)]},
),
(
"In 1951 Interest on general debt of governments ",
{"entities": [(3, 7, time),(8, 47, theme)]},
),
(
"In 2002 License taxes of governments in U.S. ",
{"entities": [(3, 7, time),(8, 36, theme),(40, 44, region)]},
),
(
"In 1972 in the United States Asbestos by census tract ",
{"entities": [(3, 7, time),(11, 28, region),(29, 37, theme),(41, 53, admin)]},
),
(
"Sales, receipts, or value of shipments of firms in 2006 in China by census tract ",
{"entities": [(0, 47, theme),(51, 55, time),(59, 64, region),(68, 80, admin)]},
),
(
"In 1999 Percent change of people enrolled in Kindergarten ",
{"entities": [(3, 7, time),(8, 57, theme)]},
),
(
"By census tract Elementary-secondary revenue from transportation programs in 1989 ",
{"entities": [(3, 15, admin),(16, 73, theme),(77, 81, time)]},
),
(
"In UK Average poverty level for household ",
{"entities": [(3, 5, region),(6, 41, theme)]},
),
(
"GDP from Utilities in 2019 by province ",
{"entities": [(0, 18, theme),(22, 26, time),(30, 38, admin)]},
),
(
"Manufacturing Corporations Quarterly After-Tax Profits in 1972 in China ",
{"entities": [(0, 54, theme),(58, 62, time),(66, 71, region)]},
),
(
"In France in 2015 by township Households with one or more people 65 years and over ",
{"entities": [(3, 9, region),(13, 17, time),(21, 29, admin),(30, 82, theme)]},
),
(
"In South Korea in 1955 by county Average hours per day spent on Household management ",
{"entities": [(3, 14, region),(18, 22, time),(26, 32, admin),(33, 84, theme)]},
),
(
"In 2017 number of people of  never married ",
{"entities": [(3, 7, time),(8, 42, theme)]},
),
(
"In 2020 Annual payroll by county in US ",
{"entities": [(3, 7, time),(8, 22, theme),(26, 32, admin),(36, 38, region)]},
),
(
"Estimated annual sales for total (excl. gasoline stations) in South Korea in 1998 ",
{"entities": [(0, 58, theme),(62, 73, region),(77, 81, time)]},
),
(
"Estimated annual sales for total (excl. motor vehicle & parts) in US by state ",
{"entities": [(0, 62, theme),(66, 68, region),(72, 77, admin)]},
),
(
"In UK Capital outlay of elementary-secondary expenditure ",
{"entities": [(3, 5, region),(6, 56, theme)]},
),
(
"Difference in race diversity by province ",
{"entities": [(0, 28, theme),(32, 40, admin)]},
),
(
"Difference in population density of All Race in 1992 in US by census tract ",
{"entities": [(0, 44, theme),(48, 52, time),(56, 58, region),(62, 74, admin)]},
),
(
"By county Average monthly housing cost in 1975 ",
{"entities": [(3, 9, admin),(10, 38, theme),(42, 46, time)]},
),
(
"In 1994 Percent change of people working more than 49 hours per week by county ",
{"entities": [(3, 7, time),(8, 68, theme),(72, 78, admin)]},
),
(
"Number of paid employees in France by county ",
{"entities": [(0, 24, theme),(28, 34, region),(38, 44, admin)]},
),
(
"By county Percent of population of American Indian and Alaska Native in 1992 in the United States ",
{"entities": [(3, 9, admin),(10, 68, theme),(72, 76, time),(80, 97, region)]},
),
(
"In 2015 Elementary-secondary revenue in Canada ",
{"entities": [(3, 7, time),(8, 36, theme),(40, 46, region)]},
),
(
"In USA number of Olympic game awards in 1991 ",
{"entities": [(3, 6, region),(7, 36, theme),(40, 44, time)]},
),
(
"In China Estimated annual sales for Furniture & home furn. Stores in 1993 by census tract ",
{"entities": [(3, 8, region),(9, 65, theme),(69, 73, time),(77, 89, admin)]},
),
(
"In South Korea in 2008 Percent change of people who are confirm to be infected by 2019-Nov Coronavirus by county ",
{"entities": [(3, 14, region),(18, 22, time),(23, 102, theme),(106, 112, admin)]},
),
(
"Land Cover Series Estimates in 1973 by province ",
{"entities": [(0, 27, theme),(31, 35, time),(39, 47, admin)]},
),
(
"Estimated annual sales for Miscellaneous store retailer in 2011 by township ",
{"entities": [(0, 55, theme),(59, 63, time),(67, 75, admin)]},
),
(
"Finance and insurance revenue in UK ",
{"entities": [(0, 29, theme),(33, 35, region)]},
),
(
"Elementary-secondary revenue from property taxes in UK in 2001 by province ",
{"entities": [(0, 48, theme),(52, 54, region),(58, 62, time),(66, 74, admin)]},
),
(
"Production workers annual hours of Bakeries and tortilla manufacturing by county ",
{"entities": [(0, 70, theme),(74, 80, admin)]},
),
(
"In UK in 2009 Current operation of governments ",
{"entities": [(3, 5, region),(9, 13, time),(14, 46, theme)]},
),
(
"Sales, receipts, or value of shipments of firms in 2001 by census tract ",
{"entities": [(0, 47, theme),(51, 55, time),(59, 71, admin)]},
),
(
"By county in UK Population density of people who are alumni of OSU ",
{"entities": [(3, 9, admin),(13, 15, region),(16, 66, theme)]},
),
(
"In US in 1973 by county Average poverty level for household ",
{"entities": [(3, 5, region),(9, 13, time),(17, 23, admin),(24, 59, theme)]},
),
(
"In UK by province Hazardous/Toxic Air Pollutants ",
{"entities": [(3, 5, region),(9, 17, admin),(18, 48, theme)]},
),
(
"In 1999 Average percent of time engaged in by womenHelping household adults by state ",
{"entities": [(3, 7, time),(8, 75, theme),(79, 84, admin)]},
),
(
"Difference in population density of  never married in US by census tract ",
{"entities": [(0, 50, theme),(54, 56, region),(60, 72, admin)]},
),
(
"In 1996 in the United States by province Lead (Pb) ",
{"entities": [(3, 7, time),(11, 28, region),(32, 40, admin),(41, 50, theme)]},
),
(
"In 1956 Estimated annual sales for total (excl. gasoline stations) in the United States ",
{"entities": [(3, 7, time),(8, 66, theme),(70, 87, region)]},
),
(
"By county in 1954 Percent of population of separated in UK ",
{"entities": [(3, 9, admin),(13, 17, time),(18, 52, theme),(56, 58, region)]},
),
(
"Difference in number of people of above age 65 by state in USA in 1969 ",
{"entities": [(0, 46, theme),(50, 55, admin),(59, 62, region),(66, 70, time)]},
),
(
"In Canada Percent change of people whose permanent teeth have been removed because of tooth decay or gum disease in 1978 ",
{"entities": [(3, 9, region),(10, 112, theme),(116, 120, time)]},
),
(
"In 1956 by census tract in the United States number of fire points ",
{"entities": [(3, 7, time),(11, 23, admin),(27, 44, region),(45, 66, theme)]},
),
(
"By township in USA homicide rate ",
{"entities": [(3, 11, admin),(15, 18, region),(19, 32, theme)]},
),
(
"Estimated annual sales for All oth. gen. merch. Stores by state in 2018 in South Korea ",
{"entities": [(0, 54, theme),(58, 63, admin),(67, 71, time),(75, 86, region)]},
),
(
"Disposable Personal Income by township ",
{"entities": [(0, 26, theme),(30, 38, admin)]},
),
(
"By township Coal Ash ",
{"entities": [(3, 11, admin),(12, 20, theme)]},
),
(
"In China Average percent of time engaged in Administrative and support activities by census tract in 1976 ",
{"entities": [(3, 8, region),(9, 81, theme),(85, 97, admin),(101, 105, time)]},
),
(
"In the United States in 2019 Salaries and wages of governments ",
{"entities": [(3, 20, region),(24, 28, time),(29, 62, theme)]},
),
(
"By state Direct expenditure of governments in US ",
{"entities": [(3, 8, admin),(9, 42, theme),(46, 48, region)]},
),
(
"In China Lending Rate by census tract in 1984 ",
{"entities": [(3, 8, region),(9, 21, theme),(25, 37, admin),(41, 45, time)]},
),
(
"In France by township Production workers annual hours of Navigational, measuring, electromedical, and control instruments manufacturing ",
{"entities": [(3, 9, region),(13, 21, admin),(22, 135, theme)]},
),
(
"Estimated annual sales for Warehouse clubs & supercenters by county ",
{"entities": [(0, 57, theme),(61, 67, admin)]},
),
(
"By county Elementary-secondary revenue from local sources in UK ",
{"entities": [(3, 9, admin),(10, 57, theme),(61, 63, region)]},
),
(
"By county sales of merchant wholesalers ",
{"entities": [(3, 9, admin),(10, 39, theme)]},
),
(
"Estimated annual sales for Family clothing stores in 1978 in South Korea by township ",
{"entities": [(0, 49, theme),(53, 57, time),(61, 72, region),(76, 84, admin)]},
),
(
"In USA number of people of people who changed the job in the past one year ",
{"entities": [(3, 6, region),(7, 74, theme)]},
),
(
"In 1990 in France by county Foreign Direct Investment ",
{"entities": [(3, 7, time),(11, 17, region),(21, 27, admin),(28, 53, theme)]},
),
(
"In France by province Annual payroll of Apparel accessories and other apparel manufacturing ",
{"entities": [(3, 9, region),(13, 21, admin),(22, 91, theme)]},
),
(
"Population density of people enrolled in Kindergarten by census tract in South Korea in 1983 ",
{"entities": [(0, 53, theme),(57, 69, admin),(73, 84, region),(88, 92, time)]},
),
(
"In 1995 Production workers annual hours of Other miscellaneous manufacturing by county ",
{"entities": [(3, 7, time),(8, 76, theme),(80, 86, admin)]},
),
(
"By province Elementary-secondary revenue from transportation programs ",
{"entities": [(3, 11, admin),(12, 69, theme)]},
),
(
"By state in 1990 Average monthly housing cost as percentage of income ",
{"entities": [(3, 8, admin),(12, 16, time),(17, 69, theme)]},
),
(
"In 1974 Elementary-secondary revenue from federal sources ",
{"entities": [(3, 7, time),(8, 57, theme)]},
),
(
"By province in 1993 Part Time Employment ",
{"entities": [(3, 11, admin),(15, 19, time),(20, 40, theme)]},
),
(
"In 2008 in US by state difference in population density of people with a bachelor's degree or higher ",
{"entities": [(3, 7, time),(11, 13, region),(17, 22, admin),(23, 100, theme)]},
),
(
"In USA area of Unconsolidated Shore (km2) ",
{"entities": [(3, 6, region),(7, 41, theme)]},
),
(
"In France in 1995 number of fixed residential broadband providers ",
{"entities": [(3, 9, region),(13, 17, time),(18, 65, theme)]},
),
(
"In France by state firearm death rate ",
{"entities": [(3, 9, region),(13, 18, admin),(19, 37, theme)]},
),
(
"In the United States by census tract Family households with own children of the householder under 18 years in 1974 ",
{"entities": [(3, 20, region),(24, 36, admin),(37, 106, theme),(110, 114, time)]},
),
(
"In 1994 Average poverty level for household ",
{"entities": [(3, 7, time),(8, 43, theme)]},
),
(
"Households with one or more people under 18 years in France in 1995 ",
{"entities": [(0, 49, theme),(53, 59, region),(63, 67, time)]},
),
(
"In 1958 Value of Construction by county ",
{"entities": [(3, 7, time),(8, 29, theme),(33, 39, admin)]},
),
(
"In 1992 by state Average household size in USA ",
{"entities": [(3, 7, time),(11, 16, admin),(17, 39, theme),(43, 46, region)]},
),
(
"By state Estimated annual sales for Shoe stores ",
{"entities": [(3, 8, admin),(9, 47, theme)]},
),
(
"Number of people of Catholic by census tract in 1977 ",
{"entities": [(0, 28, theme),(32, 44, admin),(48, 52, time)]},
),
(
"In 1980 Government Budget Value in USA by county ",
{"entities": [(3, 7, time),(8, 31, theme),(35, 38, region),(42, 48, admin)]},
),
(
"By province in USA firearm death rate ",
{"entities": [(3, 11, admin),(15, 18, region),(19, 37, theme)]},
),
(
"In 1993 in Canada by state PM2.5 ",
{"entities": [(3, 7, time),(11, 17, region),(21, 26, admin),(27, 32, theme)]},
),
(
"By province Total capital expenditures of Other nonmetallic mineral product manufacturing in France in 2015 ",
{"entities": [(3, 11, admin),(12, 89, theme),(93, 99, region),(103, 107, time)]},
),
(
"By state estimated land cover types in Canada ",
{"entities": [(3, 8, admin),(9, 35, theme),(39, 45, region)]},
),
(
"In 1966 Lending Rate by census tract ",
{"entities": [(3, 7, time),(8, 20, theme),(24, 36, admin)]},
),
(
"In the United States in 1991 Gross Fixed Capital Formation by county ",
{"entities": [(3, 20, region),(24, 28, time),(29, 58, theme),(62, 68, admin)]},
),
(
"Elementary-secondary revenue from state sources in 1959 by township ",
{"entities": [(0, 47, theme),(51, 55, time),(59, 67, admin)]},
),
(
"In 2002 Family households with own children of the householder under 18 years by state in France ",
{"entities": [(3, 7, time),(8, 77, theme),(81, 86, admin),(90, 96, region)]},
),
(
"By province Average hours per day by men spent on Helping household children with Homework in USA in 1951 ",
{"entities": [(3, 11, admin),(12, 90, theme),(94, 97, region),(101, 105, time)]},
),
(
"In 1992 Percent change of people enrolled in Nursery school, people enrolled in preschool ",
{"entities": [(3, 7, time),(8, 89, theme)]},
),
(
"Production workers annual wages of Alumina and aluminum production and processing in South Korea ",
{"entities": [(0, 81, theme),(85, 96, region)]},
),
(
"Exports By Metropolitan Area in the United States by state in 2011 ",
{"entities": [(0, 28, theme),(32, 49, region),(53, 58, admin),(62, 66, time)]},
),
(
"By county Debt at end of fiscal year of governments in 1986 ",
{"entities": [(3, 9, admin),(10, 51, theme),(55, 59, time)]},
),
(
"In China in 1980 by county Annual payroll ",
{"entities": [(3, 8, region),(12, 16, time),(20, 26, admin),(27, 41, theme)]},
),
(
"By province in 1997 Percent of population of above age 65 in China ",
{"entities": [(3, 11, admin),(15, 19, time),(20, 57, theme),(61, 66, region)]},
),
(
"In South Korea Elementary-secondary revenue from vocational programs in 1968 ",
{"entities": [(3, 14, region),(15, 68, theme),(72, 76, time)]},
),
(
"In the United States Average hours per day spent on Homework and research ",
{"entities": [(3, 20, region),(21, 73, theme)]},
),
(
"Production workers average for year of Computer and peripheral equipment manufacturing by township in 1953 in South Korea ",
{"entities": [(0, 86, theme),(90, 98, admin),(102, 106, time),(110, 121, region)]},
),
(
"In 2013 by province area of open water (km2) in UK ",
{"entities": [(3, 7, time),(11, 19, admin),(20, 44, theme),(48, 50, region)]},
),
(
"By state in USA in 2003 Total capital expenditures of Fabric mills ",
{"entities": [(3, 8, admin),(12, 15, region),(19, 23, time),(24, 66, theme)]},
),
(
"In US by state Intergovernmental expenditure of governments ",
{"entities": [(3, 5, region),(9, 14, admin),(15, 59, theme)]},
),
(
"By township Average percent of time engaged in by womenArts and entertainment (other than sports) in France ",
{"entities": [(3, 11, admin),(12, 97, theme),(101, 107, region)]},
),
(
"By census tract in China in 1961 number of people of people working more than 49 hours per week ",
{"entities": [(3, 15, admin),(19, 24, region),(28, 32, time),(33, 95, theme)]},
),
(
"Households with female householder, no husband present, family in 1998 ",
{"entities": [(0, 62, theme),(66, 70, time)]},
),
(
"Current spending of elementary-secondary expenditure in Canada in 2006 by township ",
{"entities": [(0, 52, theme),(56, 62, region),(66, 70, time),(74, 82, admin)]},
),
(
"In 1985 in USA by province Annual payroll of Ventilation, heating, air-conditioning, and commercial refrigeration equipment manufacturing ",
{"entities": [(3, 7, time),(11, 14, region),(18, 26, admin),(27, 137, theme)]},
),
(
"Married-couple family in US ",
{"entities": [(0, 21, theme),(25, 27, region)]},
),
(
"In Canada in 1991 Average poverty level for household ",
{"entities": [(3, 9, region),(13, 17, time),(18, 53, theme)]},
),
(
"Volatile Organic Compounds (VOCs) by census tract ",
{"entities": [(0, 33, theme),(37, 49, admin)]},
),
(
"In China in 1976 Households with householder living alone by census tract ",
{"entities": [(3, 8, region),(12, 16, time),(17, 57, theme),(61, 73, admin)]},
),
(
"Number of paid employees by census tract in 1970 in UK ",
{"entities": [(0, 24, theme),(28, 40, admin),(44, 48, time),(52, 54, region)]},
),
(
"Polychlorinated Biphenyls (PCBs) in China by county ",
{"entities": [(0, 32, theme),(36, 41, region),(45, 51, admin)]},
),
(
"Estimated annual sales for Grocery stores in 2007 ",
{"entities": [(0, 41, theme),(45, 49, time)]},
),
(
"By province Estimated annual sales for Elect. shopping & m/o houses ",
{"entities": [(3, 11, admin),(12, 67, theme)]},
),
(
"Number of libraries in 1985 in US by township ",
{"entities": [(0, 19, theme),(23, 27, time),(31, 33, region),(37, 45, admin)]},
),
(
"In 1954 by county Mixed Radiological Wastes in the United States ",
{"entities": [(3, 7, time),(11, 17, admin),(18, 43, theme),(47, 64, region)]},
),
(
"License taxes of governments in 2015 by province ",
{"entities": [(0, 28, theme),(32, 36, time),(40, 48, admin)]},
),
(
"Nesting information for birds breeding by township ",
{"entities": [(0, 38, theme),(42, 50, admin)]},
),
(
"Population density of  never married in 2010 by state in Canada ",
{"entities": [(0, 36, theme),(40, 44, time),(48, 53, admin),(57, 63, region)]},
),
(
"Average percent of time engaged in by menAppliances, tools, and toys by census tract in 1982 in U.S. ",
{"entities": [(0, 68, theme),(72, 84, admin),(88, 92, time),(96, 100, region)]},
),
(
"In 1971 in US by county Elementary-secondary revenue from local sources ",
{"entities": [(3, 7, time),(11, 13, region),(17, 23, admin),(24, 71, theme)]},
),
(
"In 1951 Estimated annual sales for Department stores ",
{"entities": [(3, 7, time),(8, 52, theme)]},
),
(
"In 1953 by census tract in the United States Total households ",
{"entities": [(3, 7, time),(11, 23, admin),(27, 44, region),(45, 61, theme)]},
),
(
"In USA by township Government Debt in 1984 ",
{"entities": [(3, 6, region),(10, 18, admin),(19, 34, theme),(38, 42, time)]},
),
(
"In 1962 in US Average percent of time engaged in by womenWatching TV ",
{"entities": [(3, 7, time),(11, 13, region),(14, 68, theme)]},
),
(
"In 2004 Production workers annual wages of Alumina and aluminum production and processing ",
{"entities": [(3, 7, time),(8, 89, theme)]},
),
(
"Percent of population of people working more than 49 hours per week by county ",
{"entities": [(0, 67, theme),(71, 77, admin)]},
),
(
"In South Korea by county Crude Oil Production in 1995 ",
{"entities": [(3, 14, region),(18, 24, admin),(25, 45, theme),(49, 53, time)]},
),
(
"In U.S. Household income by township ",
{"entities": [(3, 7, region),(8, 24, theme),(28, 36, admin)]},
),
(
"In US by province Average hours per day by women spent on Reading to and with household children ",
{"entities": [(3, 5, region),(9, 17, admin),(18, 96, theme)]},
),
(
"Area of Estuarine Aquatic Bed (km2) by township ",
{"entities": [(0, 35, theme),(39, 47, admin)]},
),
(
"In 1973 Average hours per day by women spent on Travel related to caring for and helping nonhousehold membership by province ",
{"entities": [(3, 7, time),(8, 112, theme),(116, 124, admin)]},
),
(
"In the United States Estimated annual sales for Food services & drinking places in 1966 ",
{"entities": [(3, 20, region),(21, 79, theme),(83, 87, time)]},
),
(
"By county Population density of people whose permanent teeth have been removed because of tooth decay or gum disease in USA ",
{"entities": [(3, 9, admin),(10, 116, theme),(120, 123, region)]},
),
(
"By province Estimated annual sales for Beer, wine & liquor stores in 1982 ",
{"entities": [(3, 11, admin),(12, 65, theme),(69, 73, time)]},
),
(
"Percent change of frauds by township in 1974 ",
{"entities": [(0, 24, theme),(28, 36, admin),(40, 44, time)]},
),
(
"By province Average year built ",
{"entities": [(3, 11, admin),(12, 30, theme)]},
),
(
"In UK Elementary-secondary revenue from local sources ",
{"entities": [(3, 5, region),(6, 53, theme)]},
),
(
"In USA by province Average hours per day by women spent on Caring for and helping household members ",
{"entities": [(3, 6, region),(10, 18, admin),(19, 99, theme)]},
),
(
"Population density of All Race in 2001 by census tract ",
{"entities": [(0, 30, theme),(34, 38, time),(42, 54, admin)]},
),
(
"By census tract in South Korea Total cost of materials of Electric lighting equipment manufacturing ",
{"entities": [(3, 15, admin),(19, 30, region),(31, 99, theme)]},
),
(
"By township Average hours per day by women spent on Appliances, tools, and toys in USA ",
{"entities": [(3, 11, admin),(12, 79, theme),(83, 86, region)]},
),
(
"In China by township Insurance trust revenue of governments in 2020 ",
{"entities": [(3, 8, region),(12, 20, admin),(21, 59, theme),(63, 67, time)]},
),
(
"Current charge of governments in US by county ",
{"entities": [(0, 29, theme),(33, 35, region),(39, 45, admin)]},
),
(
"In the United States Average hours per day spent on Leisure and sports by county ",
{"entities": [(3, 20, region),(21, 70, theme),(74, 80, admin)]},
),
(
"Estimated annual sales for Family clothing stores in France ",
{"entities": [(0, 49, theme),(53, 59, region)]},
),
(
"Average hours per day spent on Caring for and helping nonhousehold members in UK ",
{"entities": [(0, 74, theme),(78, 80, region)]},
),
(
"Households with female householder, no husband present, family in UK ",
{"entities": [(0, 62, theme),(66, 68, region)]},
),
(
"In 1955 Rental Vacancy Rate in France by township ",
{"entities": [(3, 7, time),(8, 27, theme),(31, 37, region),(41, 49, admin)]},
),
(
"Average hours per day by men spent on Personal activities in South Korea ",
{"entities": [(0, 57, theme),(61, 72, region)]},
),
(
"Ozone Effects in 1981 by census tract ",
{"entities": [(0, 13, theme),(17, 21, time),(25, 37, admin)]},
),
(
"In 1979 by county Average hours per day by women spent on Volunteering (organizational and civic activities) in China ",
{"entities": [(3, 7, time),(11, 17, admin),(18, 108, theme),(112, 117, region)]},
),
(
"By province in 1955 in US Total cost of materials of Glass and glass product manufacturing ",
{"entities": [(3, 11, admin),(15, 19, time),(23, 25, region),(26, 90, theme)]},
),
(
"Foreign Direct Investment by province ",
{"entities": [(0, 25, theme),(29, 37, admin)]},
),
(
"In South Korea in 2003 by province Estimated annual sales for Shoe stores ",
{"entities": [(3, 14, region),(18, 22, time),(26, 34, admin),(35, 73, theme)]},
),
(
"Average percent of time engaged in by menAttending household children events by province in the United States in 2007 ",
{"entities": [(0, 76, theme),(80, 88, admin),(92, 109, region),(113, 117, time)]},
),
(
"In 1971 Elementary-secondary revenue from state sources in Canada by county ",
{"entities": [(3, 7, time),(8, 55, theme),(59, 65, region),(69, 75, admin)]},
),
(
"Percent change of Jewish by township ",
{"entities": [(0, 24, theme),(28, 36, admin)]},
),
(
"By state Wages in Manufacturing ",
{"entities": [(3, 8, admin),(9, 31, theme)]},
),
(
"By county Average hours per day by men spent on Administrative and support activities ",
{"entities": [(3, 9, admin),(10, 85, theme)]},
),
(
"By township Intergovernmental expenditure of governments in 1963 in U.S. ",
{"entities": [(3, 11, admin),(12, 56, theme),(60, 64, time),(68, 72, region)]},
),
(
"In 1987 Age of householder ",
{"entities": [(3, 7, time),(8, 26, theme)]},
),
(
"By province Households with householder living alone ",
{"entities": [(3, 11, admin),(12, 52, theme)]},
),
(
"Elementary-secondary revenue from state sources by township ",
{"entities": [(0, 47, theme),(51, 59, admin)]},
),
(
"In 1997 Average hours per day by women spent on Activities related to household children education by state ",
{"entities": [(3, 7, time),(8, 98, theme),(102, 107, admin)]},
),
(
"Total cost of materials of Other chemical product and preparation manufacturing in South Korea ",
{"entities": [(0, 79, theme),(83, 94, region)]},
),
(
"In USA Elementary-secondary revenue ",
{"entities": [(3, 6, region),(7, 35, theme)]},
),
(
"By county Average household size ",
{"entities": [(3, 9, admin),(10, 32, theme)]},
),
(
"Annual payroll in 1955 in South Korea by township ",
{"entities": [(0, 14, theme),(18, 22, time),(26, 37, region),(41, 49, admin)]},
),
(
"Estimated annual sales for Nonstore retailers in China by census tract ",
{"entities": [(0, 45, theme),(49, 54, region),(58, 70, admin)]},
),
(
"Estimated annual sales for New car dealers in 1971 ",
{"entities": [(0, 42, theme),(46, 50, time)]},
),
(
"Average family size by township in China in 1952 ",
{"entities": [(0, 19, theme),(23, 31, admin),(35, 40, region),(44, 48, time)]},
),
(
"By township Mercury in 1961 in China ",
{"entities": [(3, 11, admin),(12, 19, theme),(23, 27, time),(31, 36, region)]},
),
(
"In US difference in number of people of black or African American in 1959 ",
{"entities": [(3, 5, region),(6, 65, theme),(69, 73, time)]},
),
(
"Total cost of materials of Miscellaneous manufacturing in US ",
{"entities": [(0, 54, theme),(58, 60, region)]},
),
(
"In France by province Average percent of time engaged in Attending sporting or recreational events ",
{"entities": [(3, 9, region),(13, 21, admin),(22, 98, theme)]},
),
(
"By township in 1993 Households with female householder, no husband present, family in US ",
{"entities": [(3, 11, admin),(15, 19, time),(20, 82, theme),(86, 88, region)]},
),
(
"In 1953 difference in population density of death among children under 5 due to pediatric cancer in UK ",
{"entities": [(3, 7, time),(8, 96, theme),(100, 102, region)]},
),
(
"Households with female householder, no husband present, family in China ",
{"entities": [(0, 62, theme),(66, 71, region)]},
),
(
"Dioxin Effects in Canada in 1991 ",
{"entities": [(0, 14, theme),(18, 24, region),(28, 32, time)]},
),
(
"In South Korea Total value of shipments and receipts for services of Cement and concrete product manufacturing in 1960 by province ",
{"entities": [(3, 14, region),(15, 110, theme),(114, 118, time),(122, 130, admin)]},
),
(
"By province Annual payroll of Apparel knitting mills in 2016 in France ",
{"entities": [(3, 11, admin),(12, 52, theme),(56, 60, time),(64, 70, region)]},
),
(
"Estimated annual sales for Grocery stores by province in South Korea ",
{"entities": [(0, 41, theme),(45, 53, admin),(57, 68, region)]},
),
(
"In Canada in 2001 New Home Sales ",
{"entities": [(3, 9, region),(13, 17, time),(18, 32, theme)]},
),
(
"Number of people of Catholic by census tract ",
{"entities": [(0, 28, theme),(32, 44, admin)]},
),
(
"In 1970 by state Average number of bedrooms of houses in UK ",
{"entities": [(3, 7, time),(11, 16, admin),(17, 53, theme),(57, 59, region)]},
),
(
"Households with one or more people 65 years and over by township in France in 1979 ",
{"entities": [(0, 52, theme),(56, 64, admin),(68, 74, region),(78, 82, time)]},
),
(
"In 1966 Specific conductance in Canada ",
{"entities": [(3, 7, time),(8, 28, theme),(32, 38, region)]},
),
(
"In 1961 measles incidence ",
{"entities": [(3, 7, time),(8, 25, theme)]},
),
(
"By township Carbon Monoxide Poisoning ",
{"entities": [(3, 11, admin),(12, 37, theme)]},
),
(
"In 1982 Household income ",
{"entities": [(3, 7, time),(8, 24, theme)]},
),
(
"In USA by province Average percent of time engaged in by womenHousehold services ",
{"entities": [(3, 6, region),(10, 18, admin),(19, 80, theme)]},
),
(
"In 2007 Estimated annual sales for acc. & tire store in Canada ",
{"entities": [(3, 7, time),(8, 52, theme),(56, 62, region)]},
),
(
"Production workers annual wages of Leather and allied product manufacturing in 1983 in USA ",
{"entities": [(0, 75, theme),(79, 83, time),(87, 90, region)]},
),
(
"In 1994 by township Number of employees of Animal slaughtering and processing ",
{"entities": [(3, 7, time),(11, 19, admin),(20, 77, theme)]},
),
(
"Imports by state in 1982 in the United States ",
{"entities": [(0, 7, theme),(11, 16, admin),(20, 24, time),(28, 45, region)]},
),
(
"Difference in number of people of Asian in Canada by province in 1981 ",
{"entities": [(0, 39, theme),(43, 49, region),(53, 61, admin),(65, 69, time)]},
),
(
"In 2015 by province Elementary-secondary revenue from parent government contributions in South Korea ",
{"entities": [(3, 7, time),(11, 19, admin),(20, 85, theme),(89, 100, region)]},
),
(
"In 2005 by province Sales, receipts, or value of shipments of firms in USA ",
{"entities": [(3, 7, time),(11, 19, admin),(20, 67, theme),(71, 74, region)]},
),
(
"In 1980 in U.S. Commodity by township ",
{"entities": [(3, 7, time),(11, 15, region),(16, 25, theme),(29, 37, admin)]},
),
(
"In UK Annual payroll by census tract in 2007 ",
{"entities": [(3, 5, region),(6, 20, theme),(24, 36, admin),(40, 44, time)]},
),
(
"Stock Market by census tract in 2002 ",
{"entities": [(0, 12, theme),(16, 28, admin),(32, 36, time)]},
),
(
"Elementary-secondary revenue from state sources by census tract ",
{"entities": [(0, 47, theme),(51, 63, admin)]},
),
(
"In U.S. by township in 1965 Elementary-secondary revenue from special education ",
{"entities": [(3, 7, region),(11, 19, admin),(23, 27, time),(28, 79, theme)]},
),
(
"Mercury by township ",
{"entities": [(0, 7, theme),(11, 19, admin)]},
),
(
"Average hours per day spent on Reading to and with household children in U.S. by province ",
{"entities": [(0, 69, theme),(73, 77, region),(81, 89, admin)]},
),
(
"In 1980 GNP in UK ",
{"entities": [(3, 7, time),(8, 11, theme),(15, 17, region)]},
),
(
"In 1968 by township Married-couple family ",
{"entities": [(3, 7, time),(11, 19, admin),(20, 41, theme)]},
),
(
"Average hours per day by men spent on Care for animals and pets, not veterinary care by province ",
{"entities": [(0, 84, theme),(88, 96, admin)]},
),
(
"By census tract Annual payroll in the United States in 1996 ",
{"entities": [(3, 15, admin),(16, 30, theme),(34, 51, region),(55, 59, time)]},
),
(
"In 1995 Average hours per day spent on Travel related to organizational, civic, and religious activities by province ",
{"entities": [(3, 7, time),(8, 104, theme),(108, 116, admin)]},
),
(
"In 1984 in South Korea by county Average year built ",
{"entities": [(3, 7, time),(11, 22, region),(26, 32, admin),(33, 51, theme)]},
),
(
"In 2004 Number of firms ",
{"entities": [(3, 7, time),(8, 23, theme)]},
),
(
"By township in 1952 in China Elementary-secondary revenue from special education ",
{"entities": [(3, 11, admin),(15, 19, time),(23, 28, region),(29, 80, theme)]},
),
(
"In US Average poverty level for household ",
{"entities": [(3, 5, region),(6, 41, theme)]},
),
(
"Average monthly housing cost as percentage of income in 2019 ",
{"entities": [(0, 52, theme),(56, 60, time)]},
),
(
"In 2000 Estimated annual sales for Other general merch. Stores ",
{"entities": [(3, 7, time),(8, 62, theme)]},
),
(
"By province in 1957 in China Capital outlay of elementary-secondary expenditure ",
{"entities": [(3, 11, admin),(15, 19, time),(23, 28, region),(29, 79, theme)]},
),
(
"In 1952 in South Korea Non Farm Payrolls by township ",
{"entities": [(3, 7, time),(11, 22, region),(23, 40, theme),(44, 52, admin)]},
),
(
"Age of householder in 2012 ",
{"entities": [(0, 18, theme),(22, 26, time)]},
),
(
"Carbon Monoxide (CO) in U.S. in 1995 ",
{"entities": [(0, 20, theme),(24, 28, region),(32, 36, time)]},
),
(
"Average hours per day by men spent on Helping household adults by census tract in Canada in 2008 ",
{"entities": [(0, 62, theme),(66, 78, admin),(82, 88, region),(92, 96, time)]},
),
(
"Difference in population density of people enrolled in Nursery school, people enrolled in preschool in the United States ",
{"entities": [(0, 99, theme),(103, 120, region)]},
),
(
"Sale amounts of beer by township in Canada in 2002 ",
{"entities": [(0, 20, theme),(24, 32, admin),(36, 42, region),(46, 50, time)]},
),
(
"Average percent of time engaged in Indoor and outdoor maintenance, building, and cleanup activities in U.S. ",
{"entities": [(0, 99, theme),(103, 107, region)]},
),
(
"Land Cover Series Estimates in South Korea ",
{"entities": [(0, 27, theme),(31, 42, region)]},
),
(
"In South Korea Production workers annual hours of Clay product and refractory manufacturing ",
{"entities": [(3, 14, region),(15, 91, theme)]},
),
(
"By county in South Korea Percent change of  people living in poverty areas ",
{"entities": [(3, 9, admin),(13, 24, region),(25, 74, theme)]},
),
(
"In 1985 Average monthly housing cost ",
{"entities": [(3, 7, time),(8, 36, theme)]},
),
(
"In 2019 in UK difference in population density of divorced ",
{"entities": [(3, 7, time),(11, 13, region),(14, 58, theme)]},
),
(
"In 2004 number of libraries in UK by province ",
{"entities": [(3, 7, time),(8, 27, theme),(31, 33, region),(37, 45, admin)]},
),
(
"By township Exports value of firms ",
{"entities": [(3, 11, admin),(12, 34, theme)]},
),
(
"In USA Percent change of people whose permanent teeth have been removed because of tooth decay or gum disease ",
{"entities": [(3, 6, region),(7, 109, theme)]},
),
(
"In 1980 Percent change of dentist in U.S. ",
{"entities": [(3, 7, time),(8, 33, theme),(37, 41, region)]},
),
(
"Total households by township ",
{"entities": [(0, 16, theme),(20, 28, admin)]},
),
(
"Estimated annual sales for Building material & garden eq. & supplies dealers in 1976 ",
{"entities": [(0, 76, theme),(80, 84, time)]},
),
(
"In 2016 in Canada by state GDP (nominal or ppp) per capita ",
{"entities": [(3, 7, time),(11, 17, region),(21, 26, admin),(27, 58, theme)]},
),
(
"Average percent of time engaged in by womenTravel related to caring for and helping household members in 1975 ",
{"entities": [(0, 101, theme),(105, 109, time)]},
),
(
"In USA Average monthly housing cost by province ",
{"entities": [(3, 6, region),(7, 35, theme),(39, 47, admin)]},
),
(
"In 1962 in the United States flu incidence ",
{"entities": [(3, 7, time),(11, 28, region),(29, 42, theme)]},
),
(
"In France in 2001 by province Percent of population of widowed ",
{"entities": [(3, 9, region),(13, 17, time),(21, 29, admin),(30, 62, theme)]},
),
(
"In the United States Population density of All Race in 1991 by state ",
{"entities": [(3, 20, region),(21, 51, theme),(55, 59, time),(63, 68, admin)]},
),
(
"By county Estimated annual sales for Mens clothing stores in 2009 in UK ",
{"entities": [(3, 9, admin),(10, 57, theme),(61, 65, time),(69, 71, region)]},
),
(
"Number of fixed residential broadband providers in 1966 ",
{"entities": [(0, 47, theme),(51, 55, time)]},
),
(
"In Canada in 2005 Acid Rain by township ",
{"entities": [(3, 9, region),(13, 17, time),(18, 27, theme),(31, 39, admin)]},
),
(
"Production workers annual hours of Other general purpose machinery manufacturing in 1972 ",
{"entities": [(0, 80, theme),(84, 88, time)]},
),
(
"In 1995 in South Korea Average percent of time engaged in by menHomework and research ",
{"entities": [(3, 7, time),(11, 22, region),(23, 85, theme)]},
),
(
"By province Total capital expenditures of Paper manufacturing ",
{"entities": [(3, 11, admin),(12, 61, theme)]},
),
(
"Number of people of Catholic in 2010 ",
{"entities": [(0, 28, theme),(32, 36, time)]},
),
(
"Average percent of time engaged in Household and personal messages by county ",
{"entities": [(0, 66, theme),(70, 76, admin)]},
),
(
"In the United States in 1956 by province Interest Rate ",
{"entities": [(3, 20, region),(24, 28, time),(32, 40, admin),(41, 54, theme)]},
),
(
"Crude Oil Production in 1975 ",
{"entities": [(0, 20, theme),(24, 28, time)]},
),
(
"In 1983 in US Percent of population of Asian ",
{"entities": [(3, 7, time),(11, 13, region),(14, 44, theme)]},
),
(
"In U.S. Exports value of firms by census tract ",
{"entities": [(3, 7, region),(8, 30, theme),(34, 46, admin)]},
),
(
"GDP Growth Rate in 1979 ",
{"entities": [(0, 15, theme),(19, 23, time)]},
),
(
"General revenue of governments in U.S. by township ",
{"entities": [(0, 30, theme),(34, 38, region),(42, 50, admin)]},
),
(
"Exports value of firms in 2016 ",
{"entities": [(0, 22, theme),(26, 30, time)]},
),
(
"Elementary-secondary revenue from other state aid by province in France ",
{"entities": [(0, 49, theme),(53, 61, admin),(65, 71, region)]},
),
(
"Asbestos in the United States ",
{"entities": [(0, 8, theme),(12, 29, region)]},
),
(
"By township in 1973 in France Elementary-secondary revenue from federal sources ",
{"entities": [(3, 11, admin),(15, 19, time),(23, 29, region),(30, 79, theme)]},
),
(
"In 2009 Average percent of time engaged in by womenWalking, exercising, and playing with animals by province in South Korea ",
{"entities": [(3, 7, time),(8, 96, theme),(100, 108, admin),(112, 123, region)]},
),
(
"Household income in U.S. in 1997 ",
{"entities": [(0, 16, theme),(20, 24, region),(28, 32, time)]},
),
(
"By state in 1952 Average percent of time engaged in Household management in the United States ",
{"entities": [(3, 8, admin),(12, 16, time),(17, 72, theme),(76, 93, region)]},
),



(
"In 1998 by province Average percent of time engaged in by menAttending class in Canada ",
{"entities": [(3, 7, time),(11, 19, admin),(20, 76, theme),(80, 86, region)]},
),
(
"In USA Liquor stores revenue of governments ",
{"entities": [(3, 6, region),(7, 43, theme)]},
),
(
"By township number of people of people enrolled in High school (grades 9-12) in Canada in 1988 ",
{"entities": [(3, 11, admin),(12, 76, theme),(80, 86, region),(90, 94, time)]},
),
(
"Estimated annual sales for Motor vehicle & parts Dealers in USA ",
{"entities": [(0, 56, theme),(60, 63, region)]},
),
(
"Households with one or more people 65 years and over in South Korea ",
{"entities": [(0, 52, theme),(56, 67, region)]},
),
(
"By census tract in 1958 in US Utility expenditure of governments ",
{"entities": [(3, 15, admin),(19, 23, time),(27, 29, region),(30, 64, theme)]},
),
(
"Average family size in US ",
{"entities": [(0, 19, theme),(23, 25, region)]},
),
(
"In 1973 Sulfur Dioxide (SO2) by county ",
{"entities": [(3, 7, time),(8, 28, theme),(32, 38, admin)]},
),
(
"Number of fire points by province ",
{"entities": [(0, 21, theme),(25, 33, admin)]},
),
(
"Number of McDonald's in 2014 by state ",
{"entities": [(0, 20, theme),(24, 28, time),(32, 37, admin)]},
),
(
"Annual payroll by township in 1972 in China ",
{"entities": [(0, 14, theme),(18, 26, admin),(30, 34, time),(38, 43, region)]},
),
(
"Number of people of males 15 years and over in 1985 by province ",
{"entities": [(0, 43, theme),(47, 51, time),(55, 63, admin)]},
),
(
"In 1993 in UK Average percent of time engaged in by womenPlaying games by county ",
{"entities": [(3, 7, time),(11, 13, region),(14, 70, theme),(74, 80, admin)]},
),
(
"By census tract in U.S. in 1983 Population density of people enrolled in Kindergarten ",
{"entities": [(3, 15, admin),(19, 23, region),(27, 31, time),(32, 85, theme)]},
),
(
"By township in 1950 Annual payroll of Rubber product manufacturing in UK ",
{"entities": [(3, 11, admin),(15, 19, time),(20, 66, theme),(70, 72, region)]},
),
(
"Health care and social assistance revenue in France by township ",
{"entities": [(0, 41, theme),(45, 51, region),(55, 63, admin)]},
),
(
"By census tract Coal Ash in France ",
{"entities": [(3, 15, admin),(16, 24, theme),(28, 34, region)]},
),
(
"License Taxes of governments by township in 1969 in UK ",
{"entities": [(0, 28, theme),(32, 40, admin),(44, 48, time),(52, 54, region)]},
),
(
"By county in 2012 in US Radon ",
{"entities": [(3, 9, admin),(13, 17, time),(21, 23, region),(24, 29, theme)]},
),
(
"By census tract in France homicide rate ",
{"entities": [(3, 15, admin),(19, 25, region),(26, 39, theme)]},
),
(
"In UK in 2013 area of Evergreen Forest (km2) ",
{"entities": [(3, 5, region),(9, 13, time),(14, 44, theme)]},
),
(
"Average number of bedrooms of houses in 1996 ",
{"entities": [(0, 36, theme),(40, 44, time)]},
),
(
"Estimated annual sales for acc. & tire store in U.S. in 1965 ",
{"entities": [(0, 44, theme),(48, 52, region),(56, 60, time)]},
),
(
"In 1990 Average hours per day by women spent on Travel related to purchasing goods and services ",
{"entities": [(3, 7, time),(8, 95, theme)]},
),
(
"By province Coal Ash in France ",
{"entities": [(3, 11, admin),(12, 20, theme),(24, 30, region)]},
),
(
"In UK by census tract Estimated annual sales for Home furnishings stores ",
{"entities": [(3, 5, region),(9, 21, admin),(22, 72, theme)]},
),
(
"In China in 1974 by county Percent change of American Indian and Alaska Native ",
{"entities": [(3, 8, region),(12, 16, time),(20, 26, admin),(27, 78, theme)]},
),
(
"Export Prices in US by census tract ",
{"entities": [(0, 13, theme),(17, 19, region),(23, 35, admin)]},
),
(
"In China in 1969 by township Production workers annual hours of Wood product manufacturing ",
{"entities": [(3, 8, region),(12, 16, time),(20, 28, admin),(29, 90, theme)]},
),
(
"In USA difference in population density of now married, except separated ",
{"entities": [(3, 6, region),(7, 72, theme)]},
),
(
"By county in 2010 Current charge of governments in France ",
{"entities": [(3, 9, admin),(13, 17, time),(18, 47, theme),(51, 57, region)]},
),
(
"Elementary-secondary revenue from other state aid by census tract in 1965 in France ",
{"entities": [(0, 49, theme),(53, 65, admin),(69, 73, time),(77, 83, region)]},
),
(
"By township in 2018 in US Average monthly housing cost ",
{"entities": [(3, 11, admin),(15, 19, time),(23, 25, region),(26, 54, theme)]},
),
(
"In USA Current Account ",
{"entities": [(3, 6, region),(7, 22, theme)]},
),
(
"Total value of shipments and receipts for services of Pesticide, fertilizer, and other agricultural chemical manufacturing in 1982 by township ",
{"entities": [(0, 122, theme),(126, 130, time),(134, 142, admin)]},
),
(
"In Canada by state in 1993 Percent of population of people enrolled in Nursery school, people enrolled in preschool ",
{"entities": [(3, 9, region),(13, 18, admin),(22, 26, time),(27, 115, theme)]},
),
(
"Elementary-secondary revenue from federal sources in USA ",
{"entities": [(0, 49, theme),(53, 56, region)]},
),
(
"Average hours per day by women spent on Job search and interviewing in US by census tract ",
{"entities": [(0, 67, theme),(71, 73, region),(77, 89, admin)]},
),
(
"In 1959 number of people of people who are alumni of OSU by state ",
{"entities": [(3, 7, time),(8, 56, theme),(60, 65, admin)]},
),
(
"In U.S. in 2009 Employed Persons ",
{"entities": [(3, 7, region),(11, 15, time),(16, 32, theme)]},
),
(
"Total cost of materials of Cut and sew apparel manufacturing in U.S. ",
{"entities": [(0, 60, theme),(64, 68, region)]},
),
(
"In China Average monthly housing cost as percentage of income ",
{"entities": [(3, 8, region),(9, 61, theme)]},
),
(
"In 2008 Selective sales of governments by county ",
{"entities": [(3, 7, time),(8, 38, theme),(42, 48, admin)]},
),
(
"By township in US Percent of population of  never married ",
{"entities": [(3, 11, admin),(15, 17, region),(18, 57, theme)]},
),
(
"General sales of governments by township in 1962 in Canada ",
{"entities": [(0, 28, theme),(32, 40, admin),(44, 48, time),(52, 58, region)]},
),
(
"Percent of population of People who is infected by HIV by province ",
{"entities": [(0, 54, theme),(58, 66, admin)]},
),
(
"By state Consumer Price Index CPI in South Korea in 1962 ",
{"entities": [(3, 8, admin),(9, 33, theme),(37, 48, region),(52, 56, time)]},
),
(
"In 1981 in US area of Developed, Low Intensity (km2) ",
{"entities": [(3, 7, time),(11, 13, region),(14, 52, theme)]},
),
(
"By county Households with male householder, no wife present, family in 2014 ",
{"entities": [(3, 9, admin),(10, 67, theme),(71, 75, time)]},
),
(
"In Canada Number of firms in 1973 ",
{"entities": [(3, 9, region),(10, 25, theme),(29, 33, time)]},
),
(
"By state Elementary-secondary revenue from compensatory programs in UK in 1979 ",
{"entities": [(3, 8, admin),(9, 64, theme),(68, 70, region),(74, 78, time)]},
),
(
"By county Average percent of time engaged in by menGrooming in 2019 ",
{"entities": [(3, 9, admin),(10, 59, theme),(63, 67, time)]},
),
(
"By township Gross National Product ",
{"entities": [(3, 11, admin),(12, 34, theme)]},
),
(
"In USA PM10 ",
{"entities": [(3, 6, region),(7, 11, theme)]},
),
(
"In UK Households with one or more people under 18 years ",
{"entities": [(3, 5, region),(6, 55, theme)]},
),
(
"Corporate Tax Rate in South Korea ",
{"entities": [(0, 18, theme),(22, 33, region)]},
),
(
"In U.S. Agriculture exports ",
{"entities": [(3, 7, region),(8, 27, theme)]},
),
(
"Difference in population density of death among children under 5 due to pediatric cancer by township in 1983 in UK ",
{"entities": [(0, 88, theme),(92, 100, admin),(104, 108, time),(112, 114, region)]},
),
(
"In France difference in population density of Muslim ",
{"entities": [(3, 9, region),(10, 52, theme)]},
),
(
"In 1990 number of people of divorced ",
{"entities": [(3, 7, time),(8, 36, theme)]},
),
(
"By state Elementary-secondary revenue from other state aid ",
{"entities": [(3, 8, admin),(9, 58, theme)]},
),
(
"Number of people of above age 65 in 1995 ",
{"entities": [(0, 32, theme),(36, 40, time)]},
),


(
"In 2004 by census tract GDP Annual Growth Rate ",
{"entities": [(3, 7, time),(11, 23, admin),(24, 46, theme)]},
),
(
"In China Percent change of Jewish by county ",
{"entities": [(3, 8, region),(9, 33, theme),(37, 43, admin)]},
),
(
"In U.S. by township in 1995 Asbestos ",
{"entities": [(3, 7, region),(11, 19, admin),(23, 27, time),(28, 36, theme)]},
),
(
"By province Married-couple family in the United States ",
{"entities": [(3, 11, admin),(12, 33, theme),(37, 54, region)]},
),
(
"In 1961 Core Inflation Rate ",
{"entities": [(3, 7, time),(8, 27, theme)]},
),
(
"In 2001 Average poverty level for household by state in USA ",
{"entities": [(3, 7, time),(8, 43, theme),(47, 52, admin),(56, 59, region)]},
),
(
"In 1985 Percent change of above age 65 in US ",
{"entities": [(3, 7, time),(8, 38, theme),(42, 44, region)]},
),
(
"In 1999 Soil compaction caused by humans or animals by province ",
{"entities": [(3, 7, time),(8, 51, theme),(55, 63, admin)]},
),
(
"In the United States Percent change of separated by province ",
{"entities": [(3, 20, region),(21, 48, theme),(52, 60, admin)]},
),
(
"By township number of schools ",
{"entities": [(3, 11, admin),(12, 29, theme)]},
),
(
"By township Sales, receipts, or value of shipments of firms in France in 2010 ",
{"entities": [(3, 11, admin),(12, 59, theme),(63, 69, region),(73, 77, time)]},
),
(
"Export Prices in the United States in 1976 ",
{"entities": [(0, 13, theme),(17, 34, region),(38, 42, time)]},
),
(
"In 1976 Number of employees of Other miscellaneous manufacturing ",
{"entities": [(3, 7, time),(8, 64, theme)]},
),
(
"By township Average percent of time engaged in Household and personal mail and messages ",
{"entities": [(3, 11, admin),(12, 87, theme)]},
),
(
"By census tract Average family size in U.S. in 1984 ",
{"entities": [(3, 15, admin),(16, 35, theme),(39, 43, region),(47, 51, time)]},
),
(
"By state in 1963 GDP from Construction in China ",
{"entities": [(3, 8, admin),(12, 16, time),(17, 38, theme),(42, 47, region)]},
),
(
"Difference in number of people of people with elementary occupation in 1998 by county ",
{"entities": [(0, 67, theme),(71, 75, time),(79, 85, admin)]},
),
(
"In 1952 by state Mixed Radiological Wastes ",
{"entities": [(3, 7, time),(11, 16, admin),(17, 42, theme)]},
),
(
"By province in 1960 Homeowner Vacancy Rate (percent) in Canada ",
{"entities": [(3, 11, admin),(15, 19, time),(20, 52, theme),(56, 62, region)]},
),
(
"By census tract in the United States area of Deciduous Forest (km2) ",
{"entities": [(3, 15, admin),(19, 36, region),(37, 67, theme)]},
),
(
"In 1996 by state in UK Average percent of time engaged in Socializing, relaxing, and leisure ",
{"entities": [(3, 7, time),(11, 16, admin),(20, 22, region),(23, 92, theme)]},
),
(
"By province in 1993 number of people of Muslim in Canada ",
{"entities": [(3, 11, admin),(15, 19, time),(20, 46, theme),(50, 56, region)]},
),
(
"In USA Average hours per day by men spent on Caring for and helping nonhousehold children ",
{"entities": [(3, 6, region),(7, 89, theme)]},
),
(
"Indoor Air Quality in France in 1960 ",
{"entities": [(0, 18, theme),(22, 28, region),(32, 36, time)]},
),
(
"In South Korea Average hours per day by men spent on Household management by township in 1974 ",
{"entities": [(3, 14, region),(15, 73, theme),(77, 85, admin),(89, 93, time)]},
),
(
"In 2001 Average hours per day by women spent on Arts and entertainment (other than sports) ",
{"entities": [(3, 7, time),(8, 90, theme)]},
),
(
"By county in UK in 1969 Elementary-secondary revenue from special education ",
{"entities": [(3, 9, admin),(13, 15, region),(19, 23, time),(24, 75, theme)]},
),
(
"Race diversity index in 1952 ",
{"entities": [(0, 20, theme),(24, 28, time)]},
),
(
"By state in USA Average percent of time engaged in by womenActivities related to household children education ",
{"entities": [(3, 8, admin),(12, 15, region),(16, 109, theme)]},
),
(
"By county in 1991 Average percent of time engaged in Travel related to organizational, civic, and religious activities ",
{"entities": [(3, 9, admin),(13, 17, time),(18, 118, theme)]},
),
(
"By township Average hours per day spent on Medical and care services ",
{"entities": [(3, 11, admin),(12, 68, theme)]},
),
(
"In 1970 Credit Rating ",
{"entities": [(3, 7, time),(8, 21, theme)]},
),
(
"Average hours per day by women spent on Travel related to caring for and helping household members in 1975 by census tract ",
{"entities": [(0, 98, theme),(102, 106, time),(110, 122, admin)]},
),
(
"In 1980 in US Advance International Trade in Goods by province ",
{"entities": [(3, 7, time),(11, 13, region),(14, 50, theme),(54, 62, admin)]},
),
(
"By county in U.S. number of schools in 1963 ",
{"entities": [(3, 9, admin),(13, 17, region),(18, 35, theme),(39, 43, time)]},
),
(
"Average percent of time engaged in by menOrganizational, civic, and religious activities by county in 2001 ",
{"entities": [(0, 88, theme),(92, 98, admin),(102, 106, time)]},
),
(
"In 1951 by province in China Average square footage of houses ",
{"entities": [(3, 7, time),(11, 19, admin),(23, 28, region),(29, 61, theme)]},
),
(
"Number of people of All Race by township ",
{"entities": [(0, 28, theme),(32, 40, admin)]},
),
(
"In 1987 by state in South Korea Households with male householder, no wife present, family ",
{"entities": [(3, 7, time),(11, 16, admin),(20, 31, region),(32, 89, theme)]},
),
(
"In 1988 in USA by province Government Revenues ",
{"entities": [(3, 7, time),(11, 14, region),(18, 26, admin),(27, 46, theme)]},
),
(
"Estimated annual sales for Womens clothing stores by census tract ",
{"entities": [(0, 49, theme),(53, 65, admin)]},
),
(
"Total households in China by county in 2005 ",
{"entities": [(0, 16, theme),(20, 25, region),(29, 35, admin),(39, 43, time)]},
),
(
"In 1996 in U.S. Production workers average for year of Aerospace product and parts manufacturing by province ",
{"entities": [(3, 7, time),(11, 15, region),(16, 96, theme),(100, 108, admin)]},
),
(
"Average hours per day by men spent on Travel related to purchasing goods and services in UK by province in 1964 ",
{"entities": [(0, 85, theme),(89, 91, region),(95, 103, admin),(107, 111, time)]},
),
(
"Population density of Jewish by township in France ",
{"entities": [(0, 28, theme),(32, 40, admin),(44, 50, region)]},
),
(
"By census tract in 2016 gross domestic income (nominal or ppp) per capita in South Korea ",
{"entities": [(3, 15, admin),(19, 23, time),(24, 73, theme),(77, 88, region)]},
),
(
"By census tract Radon in 1959 in France ",
{"entities": [(3, 15, admin),(16, 21, theme),(25, 29, time),(33, 39, region)]},
),
(
"In France average price for honey per pound ",
{"entities": [(3, 9, region),(10, 43, theme)]},
),
(
"Dioxin Effects by census tract ",
{"entities": [(0, 14, theme),(18, 30, admin)]},
),
(
"Average percent of time engaged in by menCaring for and helping nonhousehold children in 1968 ",
{"entities": [(0, 85, theme),(89, 93, time)]},
),
(
"In 1993 in USA by census tract Exports value of firms ",
{"entities": [(3, 7, time),(11, 14, region),(18, 30, admin),(31, 53, theme)]},
),
(
"Production workers annual wages of Sugar and confectionery product manufacturing in France ",
{"entities": [(0, 80, theme),(84, 90, region)]},
),
(
"Foreign Direct Investment in US in 2019 by state ",
{"entities": [(0, 25, theme),(29, 31, region),(35, 39, time),(43, 48, admin)]},
),
(
"By province Percent of population of above age 65 ",
{"entities": [(3, 11, admin),(12, 49, theme)]},
),
(
"In 1990 accommodation revenue by township ",
{"entities": [(3, 7, time),(8, 29, theme),(33, 41, admin)]},
),
(
"By state Special Wastes ",
{"entities": [(3, 8, admin),(9, 23, theme)]},
),
(
"In France by state Average percent of time engaged in by menTravel related to caring for and helping household members in 1957 ",
{"entities": [(3, 9, region),(13, 18, admin),(19, 118, theme),(122, 126, time)]},
),
(
"In 2013 in France by township Estimated annual sales for Building material & garden eq. & supplies dealers ",
{"entities": [(3, 7, time),(11, 17, region),(21, 29, admin),(30, 106, theme)]},
),
(
"Number of firms by township ",
{"entities": [(0, 15, theme),(19, 27, admin)]},
),
(
"Area of Unconsolidated Shore (km2) in the United States ",
{"entities": [(0, 34, theme),(38, 55, region)]},
),
(
"By province in 2005 in UK Average poverty level for household ",
{"entities": [(3, 11, admin),(15, 19, time),(23, 25, region),(26, 61, theme)]},
),
(
"Number of Olympic game awards by state ",
{"entities": [(0, 29, theme),(33, 38, admin)]},
),
(
"By township difference in population density of males 15 years and over in 1999 in U.S. ",
{"entities": [(3, 11, admin),(12, 71, theme),(75, 79, time),(83, 87, region)]},
),
(
"In China Current Account to GDP by county ",
{"entities": [(3, 8, region),(9, 31, theme),(35, 41, admin)]},
),
(
"In U.S. Percent of population of people enrolled in Kindergarten by census tract ",
{"entities": [(3, 7, region),(8, 64, theme),(68, 80, admin)]},
),



(
"Taxes of governments in 2011 by census tract ",
{"entities": [(0, 20, theme),(24, 28, time),(32, 44, admin)]},
),
(
"In UK area of Cultivated Crops (km2) ",
{"entities": [(3, 5, region),(6, 36, theme)]},
),
(
"By province in South Korea difference in population density of males 15 years and over ",
{"entities": [(3, 11, admin),(15, 26, region),(27, 86, theme)]},
),
(
"Average monthly housing cost in UK ",
{"entities": [(0, 28, theme),(32, 34, region)]},
),
(
"Corporate Profits in U.S. ",
{"entities": [(0, 17, theme),(21, 25, region)]},
),
(
"Labor Force Participation Rate in 1996 by state ",
{"entities": [(0, 30, theme),(34, 38, time),(42, 47, admin)]},
),
(
"In US in 2005 by township Production workers annual wages of Manufacturing and reproducing magnetic and optical media ",
{"entities": [(3, 5, region),(9, 13, time),(17, 25, admin),(26, 117, theme)]},
),
(
"Percent change of women that were screened for breast and cervical cancer by jurisdiction by township in China in 1998 ",
{"entities": [(0, 89, theme),(93, 101, admin),(105, 110, region),(114, 118, time)]},
),
(
"By county Production workers annual wages of Food manufacturing ",
{"entities": [(3, 9, admin),(10, 63, theme)]},
),
(
"Suicide rate by county ",
{"entities": [(0, 12, theme),(16, 22, admin)]},
),
(
"By township in 2002 difference in number of people of All Race ",
{"entities": [(3, 11, admin),(15, 19, time),(20, 62, theme)]},
),
(
"Average hours per day spent on Travel related to education by province in 1953 in France ",
{"entities": [(0, 58, theme),(62, 70, admin),(74, 78, time),(82, 88, region)]},
),
(
"In 1973 in China by census tract Average hours per day by women spent on Household services ",
{"entities": [(3, 7, time),(11, 16, region),(20, 32, admin),(33, 91, theme)]},
),
(
"Capital outlay of elementary-secondary expenditure by township in France ",
{"entities": [(0, 50, theme),(54, 62, admin),(66, 72, region)]},
),
(
"In 1990 in US difference in number of people of people who are confirm to be infected by 2019-Nov Coronavirus by census tract ",
{"entities": [(3, 7, time),(11, 13, region),(14, 109, theme),(113, 125, admin)]},
),
(
"By township Family households (families) in 1990 ",
{"entities": [(3, 11, admin),(12, 40, theme),(44, 48, time)]},
),
(
"In 1985 in UK Households with one or more people under 18 years ",
{"entities": [(3, 7, time),(11, 13, region),(14, 63, theme)]},
),
(
"In 1968 difference in population density of Asian in UK by census tract ",
{"entities": [(3, 7, time),(8, 49, theme),(53, 55, region),(59, 71, admin)]},
),
(
"In USA in 1970 Average hours per day by women spent on Attending or hosting social events by province ",
{"entities": [(3, 6, region),(10, 14, time),(15, 89, theme),(93, 101, admin)]},
),
(
"In 1979 by province Foreign Trade in France ",
{"entities": [(3, 7, time),(11, 19, admin),(20, 33, theme),(37, 43, region)]},
),
(
"In 2018 difference in population density of people enrolled in Elementary school (grades 1-8) by state ",
{"entities": [(3, 7, time),(8, 93, theme),(97, 102, admin)]},
),
(
"In 1952 zinc, oxygen, and pH in seawater ",
{"entities": [(3, 7, time),(8, 40, theme)]},
),
(
"In Canada by township Household income ",
{"entities": [(3, 9, region),(13, 21, admin),(22, 38, theme)]},
),
(
"In 1978 Total expenditure of governments in U.S. ",
{"entities": [(3, 7, time),(8, 40, theme),(44, 48, region)]},
),
(
"In 2012 in U.S. Population density of now married, except separated ",
{"entities": [(3, 7, time),(11, 15, region),(16, 67, theme)]},
),
(
"Number of firms in China ",
{"entities": [(0, 15, theme),(19, 24, region)]},
),
(
"In China General revenue of governments ",
{"entities": [(3, 8, region),(9, 39, theme)]},
),
(
"In USA Elementary-secondary revenue from other state aid in 2011 ",
{"entities": [(3, 6, region),(7, 56, theme),(60, 64, time)]},
),
(
"In 1953 by township Nonfamily households in US ",
{"entities": [(3, 7, time),(11, 19, admin),(20, 40, theme),(44, 46, region)]},
),
(
"Assistance and subsidies of governments by county ",
{"entities": [(0, 39, theme),(43, 49, admin)]},
),
(
"In South Korea by county in 2001 Production workers annual wages of Leather and allied product manufacturing ",
{"entities": [(3, 14, region),(18, 24, admin),(28, 32, time),(33, 108, theme)]},
),
(
"In 2013 number of fire points ",
{"entities": [(3, 7, time),(8, 29, theme)]},
),
(
"In 1962 by census tract in France Average percent of time engaged in by womenSocializing, relaxing, and leisure ",
{"entities": [(3, 7, time),(11, 23, admin),(27, 33, region),(34, 111, theme)]},
),
(
"In 1987 in China by province Agriculture exports ",
{"entities": [(3, 7, time),(11, 16, region),(20, 28, admin),(29, 48, theme)]},
),
(
"By county number of species ",
{"entities": [(3, 9, admin),(10, 27, theme)]},
),
(
"Total value of shipments and receipts for services of Railroad rolling stock manufacturing in 1975 ",
{"entities": [(0, 90, theme),(94, 98, time)]},
),
(
"Difference in number of people of males 15 years and over in South Korea by census tract ",
{"entities": [(0, 57, theme),(61, 72, region),(76, 88, admin)]},
),
(
"Manufacturing and International Trade in Canada in 1951 by census tract ",
{"entities": [(0, 37, theme),(41, 47, region),(51, 55, time),(59, 71, admin)]},
),
(
"By census tract Average square footage of houses in 1991 ",
{"entities": [(3, 15, admin),(16, 48, theme),(52, 56, time)]},
),
(
"In 1980 Glyphosate ",
{"entities": [(3, 7, time),(8, 18, theme)]},
),
(
"Average hours per day by women spent on Volunteer activities in 1965 by county ",
{"entities": [(0, 60, theme),(64, 68, time),(72, 78, admin)]},
),
(
"Average hours per day by men spent on Household services in 1970 in China ",
{"entities": [(0, 56, theme),(60, 64, time),(68, 73, region)]},
),
(
"Advance Estimates of U.S. Retail and Food Services by county ",
{"entities": [(0, 50, theme),(54, 60, admin)]},
)
,
(
"Family households with own children of the householder under 18 years by census tract ",
{"entities": [(0, 69, theme),(73, 85, admin)]},
),
(
"Estimated annual sales for Nonstore retailers in UK ",
{"entities": [(0, 45, theme),(49, 51, region)]},
),
(
"In Canada Human Development Index by state in 1993 ",
{"entities": [(3, 9, region),(10, 33, theme),(37, 42, admin),(46, 50, time)]},
),
(
"By township Average percent of time engaged in by womenAttending household children events in 1989 ",
{"entities": [(3, 11, admin),(12, 90, theme),(94, 98, time)]},
),
(
"In U.S. in 1991 Household income ",
{"entities": [(3, 7, region),(11, 15, time),(16, 32, theme)]},
),
(
"Average hours per day by men spent on Playing with household children, not sports in U.S. in 1986 ",
{"entities": [(0, 81, theme),(85, 89, region),(93, 97, time)]},
),
(
"Production workers average for year of Chemical manufacturing by state ",
{"entities": [(0, 61, theme),(65, 70, admin)]},
),
(
"Direct expenditure of governments in UK ",
{"entities": [(0, 33, theme),(37, 39, region)]},
),
(
"By state Age of householder in 2001 in USA ",
{"entities": [(3, 8, admin),(9, 27, theme),(31, 35, time),(39, 42, region)]},
),
(
"Estimated annual sales for Electronics & appliance stores by state ",
{"entities": [(0, 57, theme),(61, 66, admin)]},
),
(
"Professional, scientific, and technical services revenue in South Korea by county ",
{"entities": [(0, 56, theme),(60, 71, region),(75, 81, admin)]},
),
(
"By census tract Average monthly housing cost as percentage of income ",
{"entities": [(3, 15, admin),(16, 68, theme)]},
),
(
"Households with one or more people 65 years and over by census tract in UK in 1983 ",
{"entities": [(0, 52, theme),(56, 68, admin),(72, 74, region),(78, 82, time)]},
),
(
"By county difference in population density of people living in slums ",
{"entities": [(3, 9, admin),(10, 68, theme)]},
),
(
"Average percent of time engaged in by womenHelping household children with Homework in 1953 in France by state ",
{"entities": [(0, 83, theme),(87, 91, time),(95, 101, region),(105, 110, admin)]},
),
(
"In USA Number of firms by province ",
{"entities": [(3, 6, region),(7, 22, theme),(26, 34, admin)]},
),
(
"By township Bromus tectorum ",
{"entities": [(3, 11, admin),(12, 27, theme)]},
),
(
"By census tract Average number of bedrooms of houses ",
{"entities": [(3, 15, admin),(16, 52, theme)]},
),
(
"Estimated land cover types in the United States ",
{"entities": [(0, 26, theme),(30, 47, region)]},
),
(
"By township Exports value of firms in 1951 ",
{"entities": [(3, 11, admin),(12, 34, theme),(38, 42, time)]},
),
(
"Median and average prices by census tract in UK in 2019 ",
{"entities": [(0, 25, theme),(29, 41, admin),(45, 47, region),(51, 55, time)]},
),
(
"In 2007 Percent change of people living in slums by state ",
{"entities": [(3, 7, time),(8, 48, theme),(52, 57, admin)]},
),
(
"In 2019 in US by county difference in number of people of Under age 18 ",
{"entities": [(3, 7, time),(11, 13, region),(17, 23, admin),(24, 70, theme)]},
),
(
"In USA Sales, receipts, or value of shipments of firms in 2020 ",
{"entities": [(3, 6, region),(7, 54, theme),(58, 62, time)]},
),
(
"Poverty rate in US in 1991 by county ",
{"entities": [(0, 12, theme),(16, 18, region),(22, 26, time),(30, 36, admin)]},
),
(
"By township Asbestos Effects in China in 1985 ",
{"entities": [(3, 11, admin),(12, 28, theme),(32, 37, region),(41, 45, time)]},
),
(
"In 1988 by province in US Exports value of firms ",
{"entities": [(3, 7, time),(11, 19, admin),(23, 25, region),(26, 48, theme)]},
),
(
"Wildfire and beetle-caused canopy mortality by province ",
{"entities": [(0, 43, theme),(47, 55, admin)]},
),
(
"In France Average poverty level for household by province in 1987 ",
{"entities": [(3, 9, region),(10, 45, theme),(49, 57, admin),(61, 65, time)]},
),
(
"In 1994 in Canada burglary per 1000 household by state ",
{"entities": [(3, 7, time),(11, 17, region),(18, 45, theme),(49, 54, admin)]},
),
(
"By state Average hours per day spent on Telephone calls (to or from) in South Korea ",
{"entities": [(3, 8, admin),(9, 68, theme),(72, 83, region)]},
),
(
"Elementary-secondary revenue from school lunch charges in 2011 by township ",
{"entities": [(0, 54, theme),(58, 62, time),(66, 74, admin)]},
),
(
"In 1967 in China Lead Effects ",
{"entities": [(3, 7, time),(11, 16, region),(17, 29, theme)]},
),
(
"By township in Canada in 1988 Foreign Exchange Reserves ",
{"entities": [(3, 11, admin),(15, 21, region),(25, 29, time),(30, 55, theme)]},
),
(
"In Canada Government Spending to GDP ",
{"entities": [(3, 9, region),(10, 36, theme)]},
),
(
"Average percent of time engaged in Travel related to education in U.S. ",
{"entities": [(0, 62, theme),(66, 70, region)]},
),
(
"In US Sales, receipts, or value of shipments of firms ",
{"entities": [(3, 5, region),(6, 53, theme)]},
),
(
"Difference in number of people of people enrolled in High school (grades 9-12) in 1957 ",
{"entities": [(0, 78, theme),(82, 86, time)]},
),
(
"By county Selective sales of governments in China ",
{"entities": [(3, 9, admin),(10, 40, theme),(44, 49, region)]},
),
(
"Estimated annual sales for total (excl. gasoline stations) in UK ",
{"entities": [(0, 58, theme),(62, 64, region)]},
),
(
"In China by township in 1952 Estimated annual sales for Other general merch. Stores ",
{"entities": [(3, 8, region),(12, 20, admin),(24, 28, time),(29, 83, theme)]},
),
(
"In 1966 area of Perennial Ice/Snow (km2) in U.S. ",
{"entities": [(3, 7, time),(8, 40, theme),(44, 48, region)]},
),
(
"Non Farm Payrolls by township in 1960 in US ",
{"entities": [(0, 17, theme),(21, 29, admin),(33, 37, time),(41, 43, region)]},
),
(
"Elementary-secondary revenue from transportation programs in China by census tract ",
{"entities": [(0, 57, theme),(61, 66, region),(70, 82, admin)]},
),
(
"In 1995 Mold ",
{"entities": [(3, 7, time),(8, 12, theme)]},
),
(
"Number of species in 1994 by township ",
{"entities": [(0, 17, theme),(21, 25, time),(29, 37, admin)]},
),
(
"In 1984 number of cell phones per 100 person ",
{"entities": [(3, 7, time),(8, 44, theme)]},
),
(
"In 2008 by county in US Wood Burning Appliances ",
{"entities": [(3, 7, time),(11, 17, admin),(21, 23, region),(24, 47, theme)]},
),
(
"By state Average hours per day spent on Participating in religious practices ",
{"entities": [(3, 8, admin),(9, 76, theme)]},
),
(
"Difference in number of people of Hispanic or Latino Origin by township ",
{"entities": [(0, 59, theme),(63, 71, admin)]},
),
(
"In China Annual payroll of Resin, synthetic rubber, and artificial synthetic fibers and filaments manufacturing in 1991 ",
{"entities": [(3, 8, region),(9, 111, theme),(115, 119, time)]},
),
(
"In 1991 in Canada by province Elementary-secondary revenue from state sources ",
{"entities": [(3, 7, time),(11, 17, region),(21, 29, admin),(30, 77, theme)]},
),
(
"Pharmaceutical hazardous wastes in 1966 ",
{"entities": [(0, 31, theme),(35, 39, time)]},
),
(
"By township in Canada in 1980 Interbank Rate ",
{"entities": [(3, 11, admin),(15, 21, region),(25, 29, time),(30, 44, theme)]},
),
(
"Capital outlay of elementary-secondary expenditure in 2010 by province ",
{"entities": [(0, 50, theme),(54, 58, time),(62, 70, admin)]},
),
(
"Annual payroll by census tract in U.S. in 2009 ",
{"entities": [(0, 14, theme),(18, 30, admin),(34, 38, region),(42, 46, time)]},
),
(
"In US area of Deciduous Forest (km2) ",
{"entities": [(3, 5, region),(6, 36, theme)]},
),
(
"In US NSF funding for \"Catalogue\" in 2011 by census tract ",
{"entities": [(3, 5, region),(6, 33, theme),(37, 41, time),(45, 57, admin)]},
),
(
"By province in Canada Average hours per day by women spent on Homework and research in 1959 ",
{"entities": [(3, 11, admin),(15, 21, region),(22, 83, theme),(87, 91, time)]},
),
(
"By census tract number of libraries in 2009 in China ",
{"entities": [(3, 15, admin),(16, 35, theme),(39, 43, time),(47, 52, region)]},
),
(
"Mercury Effects by census tract in 1968 in USA ",
{"entities": [(0, 15, theme),(19, 31, admin),(35, 39, time),(43, 46, region)]},
),
(
"Average monthly housing cost in Canada ",
{"entities": [(0, 28, theme),(32, 38, region)]},
),
(
"Average hours per day by women spent on Personal activities in 1983 ",
{"entities": [(0, 59, theme),(63, 67, time)]},
),
(
"In China by state General revenue of governments ",
{"entities": [(3, 8, region),(12, 17, admin),(18, 48, theme)]},
),
(
"In 1982 by township number of people of Christian ",
{"entities": [(3, 7, time),(11, 19, admin),(20, 49, theme)]},
),
(
"In 2020 by census tract Asbestos Effects ",
{"entities": [(3, 7, time),(11, 23, admin),(24, 40, theme)]},
),
(
"Percent of houses with annual income of $300,000 and over by census tract in China ",
{"entities": [(0, 57, theme),(61, 73, admin),(77, 82, region)]},
),
(
"In UK real estate and rental and leasing revenue in 1998 by state ",
{"entities": [(3, 5, region),(6, 48, theme),(52, 56, time),(60, 65, admin)]},
),
(
"By township Income Taxes of governments in 2002 in France ",
{"entities": [(3, 11, admin),(12, 39, theme),(43, 47, time),(51, 57, region)]},
),
(
"By state Pharmaceutical hazardous wastes in 1982 in US ",
{"entities": [(3, 8, admin),(9, 40, theme),(44, 48, time),(52, 54, region)]},
),
(
"By county number of academic articles published in France in 1993 ",
{"entities": [(3, 9, admin),(10, 47, theme),(51, 57, region),(61, 65, time)]},
),
(
"By census tract Mercury in USA in 1995 ",
{"entities": [(3, 15, admin),(16, 23, theme),(27, 30, region),(34, 38, time)]},
),
(
"Average monthly housing cost in USA ",
{"entities": [(0, 28, theme),(32, 35, region)]},
),
(
"In U.S. in 2016 Acid Rain by township ",
{"entities": [(3, 7, region),(11, 15, time),(16, 25, theme),(29, 37, admin)]},
),
(
"In the United States in 1979 area of Perennial Ice/Snow (km2) by county ",
{"entities": [(3, 20, region),(24, 28, time),(29, 61, theme),(65, 71, admin)]},
),
(
"By census tract Percent of population of per Walmart store ",
{"entities": [(3, 15, admin),(16, 58, theme)]},
),
(
"In South Korea in 2016 Total capital expenditures of Other textile product mills ",
{"entities": [(3, 14, region),(18, 22, time),(23, 80, theme)]},
),
(
"Average year built in 1966 ",
{"entities": [(0, 18, theme),(22, 26, time)]},
),
(
"Estimated annual sales for Pharmacies & drug stores in 1960 ",
{"entities": [(0, 51, theme),(55, 59, time)]},
),
(
"By state in 2000 percent of farms with female principal operator ",
{"entities": [(3, 8, admin),(12, 16, time),(17, 64, theme)]},
),
(
"Production workers average for year of Household appliance manufacturing in South Korea by state in 1965 ",
{"entities": [(0, 72, theme),(76, 87, region),(91, 96, admin),(100, 104, time)]},
),
(
"Average percent of time engaged in by womenOther income-generating activities in 1980 in U.S. ",
{"entities": [(0, 77, theme),(81, 85, time),(89, 93, region)]},
),
(
"By township adult obesity ",
{"entities": [(3, 11, admin),(12, 25, theme)]},
),
(
"In 2009 in the United States Total value of shipments and receipts for services of Metalworking machinery manufacturing by province ",
{"entities": [(3, 7, time),(11, 28, region),(29, 119, theme),(123, 131, admin)]},
),
(
"By township Average percent of time engaged in Health-related self care ",
{"entities": [(3, 11, admin),(12, 71, theme)]},
),
(
"In 1975 Green Chemistry by census tract in France ",
{"entities": [(3, 7, time),(8, 23, theme),(27, 39, admin),(43, 49, region)]},
),
(
"Average poverty level for household in South Korea ",
{"entities": [(0, 35, theme),(39, 50, region)]},
),
(
"By province Total Housing Inventory in U.S. ",
{"entities": [(3, 11, admin),(12, 35, theme),(39, 43, region)]},
),
(
"Average percent of time engaged in by menLaundry in 1983 in France ",
{"entities": [(0, 48, theme),(52, 56, time),(60, 66, region)]},
),
(
"Population density of People who is infected by HIV in 2019 ",
{"entities": [(0, 51, theme),(55, 59, time)]},
),
(
"In 1963 number of people of  never married ",
{"entities": [(3, 7, time),(8, 42, theme)]},
),
(
"Radiocarbon dating of deep-sea (500 m to 700 m) black corals in South Korea ",
{"entities": [(0, 60, theme),(64, 75, region)]},
),
(
"In USA in 1988 Households with one or more people under 18 years by province ",
{"entities": [(3, 6, region),(10, 14, time),(15, 64, theme),(68, 76, admin)]},
),
(
"In 1970 Number of employees of Textile mills ",
{"entities": [(3, 7, time),(8, 44, theme)]},
),
(
"By township in 1953 in Canada Exports value of firms ",
{"entities": [(3, 11, admin),(15, 19, time),(23, 29, region),(30, 52, theme)]},
),
(
"In 2002 in South Korea percent of forest area ",
{"entities": [(3, 7, time),(11, 22, region),(23, 45, theme)]},
),
(
"In UK Total value of shipments and receipts for services of Textile mills in 1960 ",
{"entities": [(3, 5, region),(6, 73, theme),(77, 81, time)]},
),
(
"GDP Growth Rate in USA in 1991 ",
{"entities": [(0, 15, theme),(19, 22, region),(26, 30, time)]},
),
(
"By state Dioxin Effects in 1996 in France ",
{"entities": [(3, 8, admin),(9, 23, theme),(27, 31, time),(35, 41, region)]},
),
(
"Cathode Ray Tubes (CRTs) in 2002 ",
{"entities": [(0, 24, theme),(28, 32, time)]},
),
(
"By province Total value of shipments and receipts for services of Medical equipment and supplies manufacturing in 1967 in China ",
{"entities": [(3, 11, admin),(12, 110, theme),(114, 118, time),(122, 127, region)]},
),
(
"In the United States Number of paid employees by state ",
{"entities": [(3, 20, region),(21, 45, theme),(49, 54, admin)]},
),
(
"By county in Canada in 1992 License Taxes of governments ",
{"entities": [(3, 9, admin),(13, 19, region),(23, 27, time),(28, 56, theme)]},
),
(
"License Taxes of governments in 1953 ",
{"entities": [(0, 28, theme),(32, 36, time)]},
),
(
"By county in US Government Debt ",
{"entities": [(3, 9, admin),(13, 15, region),(16, 31, theme)]},
),
(
"In U.S. by province Interest Rate in 1969 ",
{"entities": [(3, 7, region),(11, 19, admin),(20, 33, theme),(37, 41, time)]},
),
(
"Sales, receipts, or value of shipments of firms by province in 1967 ",
{"entities": [(0, 47, theme),(51, 59, admin),(63, 67, time)]},
),
(
"Difference in population density of retailers of personal computer in 1995 ",
{"entities": [(0, 66, theme),(70, 74, time)]},
),
(
"In 2005 international trade deficit by census tract ",
{"entities": [(3, 7, time),(8, 35, theme),(39, 51, admin)]},
),
(
"Percent of population of people enrolled in Kindergarten in USA in 1982 by province ",
{"entities": [(0, 56, theme),(60, 63, region),(67, 71, time),(75, 83, admin)]},
),
(
"By state in 1977 Percent change of black or African American ",
{"entities": [(3, 8, admin),(12, 16, time),(17, 60, theme)]},
),
(
"In US GDP from Utilities in 1978 ",
{"entities": [(3, 5, region),(6, 24, theme),(28, 32, time)]},
),
(
"Soil erosion in 1997 ",
{"entities": [(0, 12, theme),(16, 20, time)]},
),
(
"In 1963 sales of merchant wholesalers ",
{"entities": [(3, 7, time),(8, 37, theme)]},
),
(
"Number of people of people enrolled in Kindergarten by county ",
{"entities": [(0, 51, theme),(55, 61, admin)]},
),
(
"In US in 1997 by township Percent of population of White ",
{"entities": [(3, 5, region),(9, 13, time),(17, 25, admin),(26, 56, theme)]},
),
(
"By province in France Estimated annual sales for Furniture & home furn. Stores in 1959 ",
{"entities": [(3, 11, admin),(15, 21, region),(22, 78, theme),(82, 86, time)]},
),
(
"Average hours per day by men spent on Storing interior household items, including food in US by state in 1959 ",
{"entities": [(0, 86, theme),(90, 92, region),(96, 101, admin),(105, 109, time)]},
),
(
"Nitrogen Dioxide (NO2) in the United States by state in 2009 ",
{"entities": [(0, 22, theme),(26, 43, region),(47, 52, admin),(56, 60, time)]},
),
(
"By state Elementary-secondary revenue from local government ",
{"entities": [(3, 8, admin),(9, 59, theme)]},
),
(
"In 2002 Average percent of time engaged in by menAppliances, tools, and toys ",
{"entities": [(3, 7, time),(8, 76, theme)]},
),
(
"In 1998 difference in number of people of retailers of personal computer in U.S. by state ",
{"entities": [(3, 7, time),(8, 72, theme),(76, 80, region),(84, 89, admin)]},
),
(
"In 1977 Gross profit of companies in Canada ",
{"entities": [(3, 7, time),(8, 33, theme),(37, 43, region)]},
),
(
"Households with householder living alone by county ",
{"entities": [(0, 40, theme),(44, 50, admin)]},
),
(
"Number of paid employees in 1973 ",
{"entities": [(0, 24, theme),(28, 32, time)]},
),
(
"By county Family households (families) ",
{"entities": [(3, 9, admin),(10, 38, theme)]},
),
(
"In the United States Total expenditure of governments ",
{"entities": [(3, 20, region),(21, 53, theme)]},
),
(
"Imports by province in UK ",
{"entities": [(0, 7, theme),(11, 19, admin),(23, 25, region)]},
),
(
"Dioxin Effects by census tract in South Korea in 2000 ",
{"entities": [(0, 14, theme),(18, 30, admin),(34, 45, region),(49, 53, time)]},
),
(
"Elementary-secondary revenue from general formula assistance by county in U.S. in 1960 ",
{"entities": [(0, 60, theme),(64, 70, admin),(74, 78, region),(82, 86, time)]},
),
(
"By census tract in 2019 Population density of people enrolled in High school (grades 9-12) in U.S. ",
{"entities": [(3, 15, admin),(19, 23, time),(24, 90, theme),(94, 98, region)]},
),
(
"Total households in U.S. ",
{"entities": [(0, 16, theme),(20, 24, region)]},
),
(
"In Canada Foreign Trade ",
{"entities": [(3, 9, region),(10, 23, theme)]},
),
(
"In U.S. in 1966 by state Average household size ",
{"entities": [(3, 7, region),(11, 15, time),(19, 24, admin),(25, 47, theme)]},
),
(
"Coal Ash in U.S. in 1978 ",
{"entities": [(0, 8, theme),(12, 16, region),(20, 24, time)]},
),
(
"In 1962 Total revenue of governments by province in Canada ",
{"entities": [(3, 7, time),(8, 36, theme),(40, 48, admin),(52, 58, region)]},
),
(
"Annual payroll in 1962 in the United States ",
{"entities": [(0, 14, theme),(18, 22, time),(26, 43, region)]},
),
(
"In U.S. by census tract number of people of people living in slums ",
{"entities": [(3, 7, region),(11, 23, admin),(24, 66, theme)]},
),
(
"In 1984 number of pedestrian accidents ",
{"entities": [(3, 7, time),(8, 38, theme)]},
),
(
"By state GDP from Construction ",
{"entities": [(3, 8, admin),(9, 30, theme)]},
),
(
"In 2000 by census tract Unemployed Persons ",
{"entities": [(3, 7, time),(11, 23, admin),(24, 42, theme)]},
),
(
"By state in 1955 International Trade in Goods in the United States ",
{"entities": [(3, 8, admin),(12, 16, time),(17, 45, theme),(49, 66, region)]},
),
(
"By census tract Production workers annual wages of Beverage and tobacco product manufacturing in 2017 in China ",
{"entities": [(3, 15, admin),(16, 93, theme),(97, 101, time),(105, 110, region)]},
),
(
"In 1970 area of Palustrine Emergent Wetland (Persistent) (km2) ",
{"entities": [(3, 7, time),(8, 62, theme)]},
),
(
"In U.S. by province Population density of People who is infected by HIV ",
{"entities": [(3, 7, region),(11, 19, admin),(20, 71, theme)]},
),
(
"In 1992 Estimated annual sales for acc. & tire store in Canada ",
{"entities": [(3, 7, time),(8, 52, theme),(56, 62, region)]},
),
(
"Difference in population density of who believe climate change by township ",
{"entities": [(0, 62, theme),(66, 74, admin)]},
),
(
"By township in the United States Number of paid employees in 1999 ",
{"entities": [(3, 11, admin),(15, 32, region),(33, 57, theme),(61, 65, time)]},
),
(
"In USA in 1950 Insurance trust expenditure of governments by state ",
{"entities": [(3, 6, region),(10, 14, time),(15, 57, theme),(61, 66, admin)]},
),
(
"In the United States Household income in 1984 ",
{"entities": [(3, 20, region),(21, 37, theme),(41, 45, time)]},
),
(
"Estimated annual sales for total (excl. motor vehicle & parts) in 1971 in UK by county ",
{"entities": [(0, 62, theme),(66, 70, time),(74, 76, region),(80, 86, admin)]},
),
(
"In 2007 area of Unconsolidated Shore (km2) ",
{"entities": [(3, 7, time),(8, 42, theme)]},
),
(
"Universal Waste in 1994 ",
{"entities": [(0, 15, theme),(19, 23, time)]},
),
(
"In U.S. by census tract Liquor stores expenditure of governments ",
{"entities": [(3, 7, region),(11, 23, admin),(24, 64, theme)]},
),
(
"Average hours per day by men spent on Attending class by township ",
{"entities": [(0, 53, theme),(57, 65, admin)]},
),
(
"In U.S. in 2017 difference in population density of separated ",
{"entities": [(3, 7, region),(11, 15, time),(16, 61, theme)]},
),
(
"By county Population density of Under age 18 ",
{"entities": [(3, 9, admin),(10, 44, theme)]},
),
(
"In the United States License Taxes of governments in 1980 ",
{"entities": [(3, 20, region),(21, 49, theme),(53, 57, time)]},
),
(
"In South Korea Capital Flows by county in 1981 ",
{"entities": [(3, 14, region),(15, 28, theme),(32, 38, admin),(42, 46, time)]},
),
(
"In U.S. in 1967 percent of farmland by province ",
{"entities": [(3, 7, region),(11, 15, time),(16, 35, theme),(39, 47, admin)]},
),
(
"Percent of population of people enrolled in College or graduate school in China ",
{"entities": [(0, 70, theme),(74, 79, region)]},
),
(
"In 2005 by census tract Households with female householder, no husband present, family ",
{"entities": [(3, 7, time),(11, 23, admin),(24, 86, theme)]},
),
(
"By census tract Food Waste and Recovery ",
{"entities": [(3, 15, admin),(16, 39, theme)]},
),
(
"By census tract Current spending of elementary-secondary expenditure in South Korea in 2013 ",
{"entities": [(3, 15, admin),(16, 68, theme),(72, 83, region),(87, 91, time)]},
),
(
"Number of multi-racial households in US in 1972 by census tract ",
{"entities": [(0, 33, theme),(37, 39, region),(43, 47, time),(51, 63, admin)]},
),
(
"In France by state Taxes of governments ",
{"entities": [(3, 9, region),(13, 18, admin),(19, 39, theme)]},
),
(
"Wood Burning Appliances by census tract ",
{"entities": [(0, 23, theme),(27, 39, admin)]},
),
(
"In UK Number of firms ",
{"entities": [(3, 5, region),(6, 21, theme)]},
),
(
"In South Korea by township Population density of Hispanic or Latino Origin ",
{"entities": [(3, 14, region),(18, 26, admin),(27, 74, theme)]},
),
(
"By township in the United States difference in population density of above age 65 ",
{"entities": [(3, 11, admin),(15, 32, region),(33, 81, theme)]},
),
(
"Average hours per day by men spent on Participating in performance and cultural by province ",
{"entities": [(0, 79, theme),(83, 91, admin)]},
),
(
"Households with female householder, no husband present, family in China ",
{"entities": [(0, 62, theme),(66, 71, region)]},
),
(
"In US difference in population density of people who are alumni of OSU ",
{"entities": [(3, 5, region),(6, 70, theme)]},
),
(
"Area of Palustrine Emergent Wetland (Persistent) (km2) by census tract in 1973 ",
{"entities": [(0, 54, theme),(58, 70, admin),(74, 78, time)]},
),
(
"In US number of people of people with a bachelor's degree or higher in 1953 by province ",
{"entities": [(3, 5, region),(6, 67, theme),(71, 75, time),(79, 87, admin)]},
),
(
"Average hours per day by men spent on Civic obligations and participation by province in 1966 ",
{"entities": [(0, 73, theme),(77, 85, admin),(89, 93, time)]},
),
(
"In 1978 by township Percent change of Under age 18 ",
{"entities": [(3, 7, time),(11, 19, admin),(20, 50, theme)]},
),
(
"In U.S. Gold Reserves in 1964 ",
{"entities": [(3, 7, region),(8, 21, theme),(25, 29, time)]},
),
(
"Balance of Trade in US by province ",
{"entities": [(0, 16, theme),(20, 22, region),(26, 34, admin)]},
),
(
"Diabetes rate in 1981 ",
{"entities": [(0, 13, theme),(17, 21, time)]},
),
(
"In 1984 Elementary-secondary revenue from local sources in USA ",
{"entities": [(3, 7, time),(8, 55, theme),(59, 62, region)]},
),
(
"In 2001 Homeownership rates in U.S. by county ",
{"entities": [(3, 7, time),(8, 27, theme),(31, 35, region),(39, 45, admin)]},
),
(
"In South Korea Number of firms in 2017 ",
{"entities": [(3, 14, region),(15, 30, theme),(34, 38, time)]},
),
(
"By census tract in the United States in 1964 Capital outlay of elementary-secondary expenditure ",
{"entities": [(3, 15, admin),(19, 36, region),(40, 44, time),(45, 95, theme)]},
),
(
"Consumer Confidence in 2004 by state in the United States ",
{"entities": [(0, 19, theme),(23, 27, time),(31, 36, admin),(40, 57, region)]},
),
(
"In the United States Production workers annual hours of Commercial and service industry machinery manufacturing ",
{"entities": [(3, 20, region),(21, 111, theme)]},
),
(
"In 1950 Pesticide Chemicals by township in France ",
{"entities": [(3, 7, time),(8, 27, theme),(31, 39, admin),(43, 49, region)]},
),
(
"In 1983 by township in the United States Production workers annual hours of Transportation equipment manufacturing ",
{"entities": [(3, 7, time),(11, 19, admin),(23, 40, region),(41, 114, theme)]},
),
(
"By county in 2009 in USA Dioxin Effects ",
{"entities": [(3, 9, admin),(13, 17, time),(21, 24, region),(25, 39, theme)]},
),
(
"By census tract in 1978 Percent change of  people living in poverty areas ",
{"entities": [(3, 15, admin),(19, 23, time),(24, 73, theme)]},
),
(
"In 1960 Business Applications ",
{"entities": [(3, 7, time),(8, 29, theme)]},
),
(
"In UK number of academic articles published by province ",
{"entities": [(3, 5, region),(6, 43, theme),(47, 55, admin)]},
),
(
"Households with female householder, no husband present, family in 1951 ",
{"entities": [(0, 62, theme),(66, 70, time)]},
),
(
"In US Elementary-secondary revenue from special education in 2007 by province ",
{"entities": [(3, 5, region),(6, 57, theme),(61, 65, time),(69, 77, admin)]},
),
(
"In France difference in population density of Asian in 1993 ",
{"entities": [(3, 9, region),(10, 51, theme),(55, 59, time)]},
),
(
"Zinc, oxygen, and pH in seawater in China ",
{"entities": [(0, 32, theme),(36, 41, region)]},
),
(
"Area of Sedge Herbaceous (km2) in 1971 ",
{"entities": [(0, 30, theme),(34, 38, time)]},
),
(
"In France by state Utility expenditure of governments ",
{"entities": [(3, 9, region),(13, 18, admin),(19, 53, theme)]},
),
(
"Difference in population density of retailers of personal computer in Canada ",
{"entities": [(0, 66, theme),(70, 76, region)]},
),
(
"In US by township in 1957 Current Account to GDP ",
{"entities": [(3, 5, region),(9, 17, admin),(21, 25, time),(26, 48, theme)]},
),
(
"By province GDP Constant Prices in 2008 ",
{"entities": [(3, 11, admin),(12, 31, theme),(35, 39, time)]},
),
(
"By census tract Elementary-secondary revenue from other state aid in 2010 ",
{"entities": [(3, 15, admin),(16, 65, theme),(69, 73, time)]},
),
(
"In UK Population density of people who are alumni of OSU in 2011 by state ",
{"entities": [(3, 5, region),(6, 56, theme),(60, 64, time),(68, 73, admin)]},
),
(
"Percent of farms with female principal operator by county in 1998 in UK ",
{"entities": [(0, 47, theme),(51, 57, admin),(61, 65, time),(69, 71, region)]},
),
(
"Difference in number of people of retailers of personal computer in 1980 by province ",
{"entities": [(0, 64, theme),(68, 72, time),(76, 84, admin)]},
),
(
"In 2001 Current spending of elementary-secondary expenditure by state in UK ",
{"entities": [(3, 7, time),(8, 60, theme),(64, 69, admin),(73, 75, region)]},
),
(
"By county in USA in 1969 Average hours per day by women spent on Arts and entertainment (other than sports) ",
{"entities": [(3, 9, admin),(13, 16, region),(20, 24, time),(25, 107, theme)]},
),
(
"In 1950 Percent change of Muslim ",
{"entities": [(3, 7, time),(8, 32, theme)]},
),
(
"In U.S. by census tract in 1976 Estimated annual sales for total (excl. motor vehicle & parts) ",
{"entities": [(3, 7, region),(11, 23, admin),(27, 31, time),(32, 94, theme)]},
),
(
"In Canada by census tract Average percent of time engaged in by womenParticipating in performance and cultural in 1970 ",
{"entities": [(3, 9, region),(13, 25, admin),(26, 110, theme),(114, 118, time)]},
),
(
"By state Average poverty level for household in France ",
{"entities": [(3, 8, admin),(9, 44, theme),(48, 54, region)]},
),
(
"By province number of people of Muslim ",
{"entities": [(3, 11, admin),(12, 38, theme)]},
),
(
"By county in 2018 area of Developed, High Intensity (km2) ",
{"entities": [(3, 9, admin),(13, 17, time),(18, 57, theme)]},
),
(
"Total cost of materials of Semiconductor and other electronic component manufacturing by state ",
{"entities": [(0, 85, theme),(89, 94, admin)]},
),
(
"Sea-Floor Sediment in UK by township ",
{"entities": [(0, 18, theme),(22, 24, region),(28, 36, admin)]},
),
(
"Households with one or more people under 18 years in South Korea in 1999 by county ",
{"entities": [(0, 49, theme),(53, 64, region),(68, 72, time),(76, 82, admin)]},
),
(
"In 2006 Percent change of people below poverty level in Canada ",
{"entities": [(3, 7, time),(8, 52, theme),(56, 62, region)]},
),
(
"By census tract Number of firms ",
{"entities": [(3, 15, admin),(16, 31, theme)]},
),
(
"In 1979 by township in Canada number of people of Catholic ",
{"entities": [(3, 7, time),(11, 19, admin),(23, 29, region),(30, 58, theme)]},
),
(
"Money Supply in 1996 in the United States ",
{"entities": [(0, 12, theme),(16, 20, time),(24, 41, region)]},
),
(
"By province Average hours per day by men spent on Travel related to telephone calls ",
{"entities": [(3, 11, admin),(12, 83, theme)]},
),
(
"Family households (families) in U.S. by state in 2012 ",
{"entities": [(0, 28, theme),(32, 36, region),(40, 45, admin),(49, 53, time)]},
),
(
"Average hours per day by women spent on Travel related to education in 2020 by county ",
{"entities": [(0, 67, theme),(71, 75, time),(79, 85, admin)]},
),
(
"Area of Cultivated Crops (km2) by state ",
{"entities": [(0, 30, theme),(34, 39, admin)]},
),
(
"Average percent of time engaged in by womenHousehold services in 1992 ",
{"entities": [(0, 61, theme),(65, 69, time)]},
),
(
"In the United States Average percent of time engaged in by womenSocial service and care activities  ",
{"entities": [(3, 20, region),(21, 99, theme)]},
),
(
"By census tract in China in 1950 Elementary-secondary revenue from transportation programs ",
{"entities": [(3, 15, admin),(19, 24, region),(28, 32, time),(33, 90, theme)]},
),
(
"Average percent of time engaged in by womenMedical and care services in US ",
{"entities": [(0, 68, theme),(72, 74, region)]},
),
(
"Average year built in 1965 in U.S. ",
{"entities": [(0, 18, theme),(22, 26, time),(30, 34, region)]},
),
(
"By township difference in number of people of frauds in Canada in 1976 ",
{"entities": [(3, 11, admin),(12, 52, theme),(56, 62, region),(66, 70, time)]},
),
(
"Number of paid employees in US ",
{"entities": [(0, 24, theme),(28, 30, region)]},
),
(
"In France Average hours per day by women spent on Financial services and banking ",
{"entities": [(3, 9, region),(10, 80, theme)]},
),
(
"In 1973 difference in number of people of widowed by province ",
{"entities": [(3, 7, time),(8, 49, theme),(53, 61, admin)]},
),
(
"Average percent of time engaged in by womenHousehold and personal messages in Canada ",
{"entities": [(0, 74, theme),(78, 84, region)]},
),
(
"Current Account to GDP in USA ",
{"entities": [(0, 22, theme),(26, 29, region)]},
),
(
"In U.S. Average number of bedrooms of houses in 2012 ",
{"entities": [(3, 7, region),(8, 44, theme),(48, 52, time)]},
),
(
"Estimated annual sales for Building mat. & sup. dealers in 1958 ",
{"entities": [(0, 55, theme),(59, 63, time)]},
),
(
"Area of Developed, Open Space (km2) by census tract ",
{"entities": [(0, 35, theme),(39, 51, admin)]},
),
(
"Average hours per day spent on Participating in sports, exercise, and recreation in South Korea ",
{"entities": [(0, 80, theme),(84, 95, region)]},
),
(
"By township in U.S. in 1981 Glyphosate ",
{"entities": [(3, 11, admin),(15, 19, region),(23, 27, time),(28, 38, theme)]},
),
(
"In China Particulate Matter (PM) by township in 1958 ",
{"entities": [(3, 8, region),(9, 32, theme),(36, 44, admin),(48, 52, time)]},
),
(
"In 1959 Elementary-secondary revenue from state sources ",
{"entities": [(3, 7, time),(8, 55, theme)]},
),
(
"Helicobacter pylori rate in US by county in 1962 ",
{"entities": [(0, 24, theme),(28, 30, region),(34, 40, admin),(44, 48, time)]},
),
(
"By county Estimated annual sales for Sporting goods hobby, musical instrument, & book stores in 1996 in the United States ",
{"entities": [(3, 9, admin),(10, 92, theme),(96, 100, time),(104, 121, region)]},
),
(
"In US crime rate ",
{"entities": [(3, 5, region),(6, 16, theme)]},
),
(
"In UK Estimated annual sales for Nonstore retailers in 1969 ",
{"entities": [(3, 5, region),(6, 51, theme),(55, 59, time)]},
),
(
"Estimated annual sales for Home furnishings stores in UK in 1951 ",
{"entities": [(0, 50, theme),(54, 56, region),(60, 64, time)]},
),
(
"In USA Total Housing Inventory ",
{"entities": [(3, 6, region),(7, 30, theme)]},
),
(
"By county well-being index in 2000 in US ",
{"entities": [(3, 9, admin),(10, 26, theme),(30, 34, time),(38, 40, region)]},
),
(
"In U.S. by census tract Estimated annual sales for total (excl. gasoline stations) ",
{"entities": [(3, 7, region),(11, 23, admin),(24, 82, theme)]},
),
(
"In U.S. in 1984 Population density of people enrolled in High school (grades 9-12) ",
{"entities": [(3, 7, region),(11, 15, time),(16, 82, theme)]},
),
(
"In 1994 Exports value of firms by township ",
{"entities": [(3, 7, time),(8, 30, theme),(34, 42, admin)]},
),
(
"By census tract Estimated annual sales for Warehouse clubs & supercenters ",
{"entities": [(3, 15, admin),(16, 73, theme)]},
),
(
"Average hours per day spent on Travel related to organizational, civic, and religious activities by state ",
{"entities": [(0, 96, theme),(100, 105, admin)]},
),
(
"In 2008 in UK Total revenue of governments by county ",
{"entities": [(3, 7, time),(11, 13, region),(14, 42, theme),(46, 52, admin)]},
),
(
"Area of Deciduous Forest (km2) in 1998 in USA by county ",
{"entities": [(0, 30, theme),(34, 38, time),(42, 45, region),(49, 55, admin)]},
),
(
"Estimated annual sales for Elect. shopping & m/o houses by township ",
{"entities": [(0, 55, theme),(59, 67, admin)]},
),
(
"In 1991 Average hours per day spent on Job search and interviewing ",
{"entities": [(3, 7, time),(8, 66, theme)]},
),
(
"In China food insecurity rate ",
{"entities": [(3, 8, region),(9, 29, theme)]},
),
(
"Estimated annual sales for Furniture stores in U.S. by census tract in 1972 ",
{"entities": [(0, 43, theme),(47, 51, region),(55, 67, admin),(71, 75, time)]},
),
(
"In 1955 Average year built ",
{"entities": [(3, 7, time),(8, 26, theme)]},
),
(
"In 2007 Households with male householder, no wife present, family ",
{"entities": [(3, 7, time),(8, 65, theme)]},
),
(
"In China Estimated annual sales for total (excl. motor vehicle & parts & gasoline stations) ",
{"entities": [(3, 8, region),(9, 91, theme)]},
),
(
"Percent of population of people enrolled in Kindergarten in UK by province in 2009 ",
{"entities": [(0, 56, theme),(60, 62, region),(66, 74, admin),(78, 82, time)]},
),
(
"In the United States Exports By Metropolitan Area in 1987 ",
{"entities": [(3, 20, region),(21, 49, theme),(53, 57, time)]},
),
(
"Estimated annual sales for Womens clothing stores in Canada ",
{"entities": [(0, 49, theme),(53, 59, region)]},
),
(
"GNP by census tract ",
{"entities": [(0, 3, theme),(7, 19, admin)]},
),
(
"Percent change of black or African American in 1957 ",
{"entities": [(0, 43, theme),(47, 51, time)]},
),
(
"In USA Glyphosate in 2014 by township ",
{"entities": [(3, 6, region),(7, 17, theme),(21, 25, time),(29, 37, admin)]},
),
(
"By state Intergovernmental revenue of governments in UK in 1996 ",
{"entities": [(3, 8, admin),(9, 49, theme),(53, 55, region),(59, 63, time)]},
),
(
"Average percent of time engaged in by womenWatching TV in USA by province in 1966 ",
{"entities": [(0, 54, theme),(58, 61, region),(65, 73, admin),(77, 81, time)]},
),
(
"By province Percent change of people with a bachelor's degree or higher ",
{"entities": [(3, 11, admin),(12, 71, theme)]},
),
(
"By province educational services revenue ",
{"entities": [(3, 11, admin),(12, 40, theme)]},
),
(
"By county in 1999 Mercury ",
{"entities": [(3, 9, admin),(13, 17, time),(18, 25, theme)]},
),
(
"By county Race diversity index ",
{"entities": [(3, 9, admin),(10, 30, theme)]},
),
(
"In 1953 Households with male householder, no wife present, family ",
{"entities": [(3, 7, time),(8, 65, theme)]},
),
(
"In 1980 Estimated annual sales for General merchandise stores ",
{"entities": [(3, 7, time),(8, 61, theme)]},
),
(
"Direct expenditure of governments by province in the United States ",
{"entities": [(0, 33, theme),(37, 45, admin),(49, 66, region)]},
),
(
"By county Production workers average for year of Other general purpose machinery manufacturing ",
{"entities": [(3, 9, admin),(10, 94, theme)]},
),
(
"Average percent of time engaged in Grooming in 2014 in China ",
{"entities": [(0, 43, theme),(47, 51, time),(55, 60, region)]},
),
(
"New Home Sales in 1990 by county ",
{"entities": [(0, 14, theme),(18, 22, time),(26, 32, admin)]},
),
(
"Gross National Product in France ",
{"entities": [(0, 22, theme),(26, 32, region)]},
),
(
"Total Taxes of governments in UK ",
{"entities": [(0, 26, theme),(30, 32, region)]},
),
(
"Consumer Price Index CPI by province in 2004 ",
{"entities": [(0, 24, theme),(28, 36, admin),(40, 44, time)]},
),
(
"In 2015 by census tract in China Liquor stores expenditure of governments ",
{"entities": [(3, 7, time),(11, 23, admin),(27, 32, region),(33, 73, theme)]},
),
(
"In 1974 in China Average number of bedrooms of houses by census tract ",
{"entities": [(3, 7, time),(11, 16, region),(17, 53, theme),(57, 69, admin)]},
),
(
"By county in US in 1979 difference in population density of Catholic ",
{"entities": [(3, 9, admin),(13, 15, region),(19, 23, time),(24, 68, theme)]},
),
(
"Estimated annual sales for Health & personal care stores by county ",
{"entities": [(0, 56, theme),(60, 66, admin)]},
),
(
"By county difference in population density of  never married in 1954 ",
{"entities": [(3, 9, admin),(10, 60, theme),(64, 68, time)]},
),
(
"By state Production workers annual wages of Tobacco manufacturing ",
{"entities": [(3, 8, admin),(9, 65, theme)]},
),
(
"In UK in 1992 by township suicide rate ",
{"entities": [(3, 5, region),(9, 13, time),(17, 25, admin),(26, 38, theme)]},
),
(
"In China in 2016 by state professional, scientific, and technical services revenue ",
{"entities": [(3, 8, region),(12, 16, time),(20, 25, admin),(26, 82, theme)]},
),
(
"In 2017 Assistance and subsidies of governments in UK ",
{"entities": [(3, 7, time),(8, 47, theme),(51, 53, region)]},
),
(
"By township Capital outlay of governments ",
{"entities": [(3, 11, admin),(12, 41, theme)]},
),
(
"In the United States by county Government Revenues ",
{"entities": [(3, 20, region),(24, 30, admin),(31, 50, theme)]},
),
(
"Average percent of time engaged in by menAttending religious services in China ",
{"entities": [(0, 69, theme),(73, 78, region)]},
),
(
"By township in 1996 Average hours per day by men spent on Helping nonhousehold adults in the United States ",
{"entities": [(3, 11, admin),(15, 19, time),(20, 85, theme),(89, 106, region)]},
),
(
"Direct expenditure of governments in 1994 ",
{"entities": [(0, 33, theme),(37, 41, time)]},
),
(
"In South Korea by census tract in 1984 Terms of Trade ",
{"entities": [(3, 14, region),(18, 30, admin),(34, 38, time),(39, 53, theme)]},
),
(
"Average percent of time engaged in by menCivic obligations and participation by state in 1954 in U.S. ",
{"entities": [(0, 76, theme),(80, 85, admin),(89, 93, time),(97, 101, region)]},
),
(
"Current Account to GDP by census tract ",
{"entities": [(0, 22, theme),(26, 38, admin)]},
),
(
"In China Elementary-secondary revenue ",
{"entities": [(3, 8, region),(9, 37, theme)]},
),
(
"By county in US Annual payroll ",
{"entities": [(3, 9, admin),(13, 15, region),(16, 30, theme)]},
),
(
"By township Soil erosion ",
{"entities": [(3, 11, admin),(12, 24, theme)]},
),
(
"In the United States difference in race diversity in 1974 ",
{"entities": [(3, 20, region),(21, 49, theme),(53, 57, time)]},
),
(
"In South Korea Total cost of materials of Food manufacturing ",
{"entities": [(3, 14, region),(15, 60, theme)]},
),
(
"Carbon Monoxide Poisoning by census tract in 2017 in US ",
{"entities": [(0, 25, theme),(29, 41, admin),(45, 49, time),(53, 55, region)]},
),
(
"In 1962 Production workers average for year of Hardware manufacturing,Spring and wire product manufacturing ",
{"entities": [(3, 7, time),(8, 107, theme)]},
),
(
"Number of firms by county in US ",
{"entities": [(0, 15, theme),(19, 25, admin),(29, 31, region)]},
),
(
"By state in 1955 number of pedestrian accidents in South Korea ",
{"entities": [(3, 8, admin),(12, 16, time),(17, 47, theme),(51, 62, region)]},
),
(
"By state in 2001 in South Korea Cash and security holdings of governments ",
{"entities": [(3, 8, admin),(12, 16, time),(20, 31, region),(32, 73, theme)]},
),
(
"Family households (families) in 1997 in Canada ",
{"entities": [(0, 28, theme),(32, 36, time),(40, 46, region)]},
),
(
"Current spending of elementary-secondary expenditure in 2006 ",
{"entities": [(0, 52, theme),(56, 60, time)]},
),
(
"In U.S. in 2010 Herbicide use by county ",
{"entities": [(3, 7, region),(11, 15, time),(16, 29, theme),(33, 39, admin)]},
),
(
"Per- and Polyfluoroalkyl Substances (PFAS) in U.S. in 1955 ",
{"entities": [(0, 42, theme),(46, 50, region),(54, 58, time)]},
),
(
"Production workers annual wages of Motor vehicle parts manufacturing in 1964 ",
{"entities": [(0, 68, theme),(72, 76, time)]},
),
(
"Households with one or more people under 18 years by county in France ",
{"entities": [(0, 49, theme),(53, 59, admin),(63, 69, region)]},
),
(
"By state GDP Growth Rate ",
{"entities": [(3, 8, admin),(9, 24, theme)]},
),
(
"By census tract Average percent of time engaged in by womenArts and entertainment (other than sports) ",
{"entities": [(3, 15, admin),(16, 101, theme)]},
),
(
"In France by state in 1999 Percent change of people who are alumni of OSU ",
{"entities": [(3, 9, region),(13, 18, admin),(22, 26, time),(27, 73, theme)]},
),
(
"Interest on debt of governments in 1995 by township ",
{"entities": [(0, 31, theme),(35, 39, time),(43, 51, admin)]},
),
(
"Area of Barren Land (km2) by province in 1964 in France ",
{"entities": [(0, 25, theme),(29, 37, admin),(41, 45, time),(49, 55, region)]},
),
(
"Indoor Air Quality in China in 2013 by province ",
{"entities": [(0, 18, theme),(22, 27, region),(31, 35, time),(39, 47, admin)]},
),
(
"In UK Household income ",
{"entities": [(3, 5, region),(6, 22, theme)]},
),
(
"Estimated annual sales for Furniture stores by township in 1983 ",
{"entities": [(0, 43, theme),(47, 55, admin),(59, 63, time)]},
),
(
"Average percent of time engaged in Caring for and helping household children in 1978 ",
{"entities": [(0, 76, theme),(80, 84, time)]},
),
(
"In U.S. Sales, receipts, or value of shipments of firms ",
{"entities": [(3, 7, region),(8, 55, theme)]},
),
(
"In 1956 Current Account ",
{"entities": [(3, 7, time),(8, 23, theme)]},
),
(
"In US by township difference in number of people of above age 65 ",
{"entities": [(3, 5, region),(9, 17, admin),(18, 64, theme)]},
),
(
"Interest Rate by state in Canada ",
{"entities": [(0, 13, theme),(17, 22, admin),(26, 32, region)]},
),
(
"In US Total Housing Inventory in 1963 ",
{"entities": [(3, 5, region),(6, 29, theme),(33, 37, time)]},
),
(
"By township Seasonally adjusted sales in 2018 ",
{"entities": [(3, 11, admin),(12, 37, theme),(41, 45, time)]},
),
(
"In 1978 by province area of Mixed Forest (km2) ",
{"entities": [(3, 7, time),(11, 19, admin),(20, 46, theme)]},
),
(
"By census tract International Electronic Waste (E-Waste) in 1983 in U.S. ",
{"entities": [(3, 15, admin),(16, 56, theme),(60, 64, time),(68, 72, region)]},
),
(
"Number of hospitals in 1995 ",
{"entities": [(0, 19, theme),(23, 27, time)]},
),
(
"By county Sales, receipts, or value of shipments of firms in France in 1994 ",
{"entities": [(3, 9, admin),(10, 57, theme),(61, 67, region),(71, 75, time)]},
),
(
"In 1954 Elementary-secondary revenue from vocational programs in UK ",
{"entities": [(3, 7, time),(8, 61, theme),(65, 67, region)]},
),
(
"By state in Canada number of people of above age 65 in 1951 ",
{"entities": [(3, 8, admin),(12, 18, region),(19, 51, theme),(55, 59, time)]},
),
(
"In 1997 in USA Percent change of frauds by county ",
{"entities": [(3, 7, time),(11, 14, region),(15, 39, theme),(43, 49, admin)]},
),
(
"Labor Force Participation Rate in China by county ",
{"entities": [(0, 30, theme),(34, 39, region),(43, 49, admin)]},
),
(
"Family households (families) in 1983 by province in South Korea ",
{"entities": [(0, 28, theme),(32, 36, time),(40, 48, admin),(52, 63, region)]},
),
(
"Insurance trust expenditure of governments in 1967 in U.S. by county ",
{"entities": [(0, 42, theme),(46, 50, time),(54, 58, region),(62, 68, admin)]},
),
(
"In 2005 by county difference in number of people of people working more than 49 hours per week ",
{"entities": [(3, 7, time),(11, 17, admin),(18, 94, theme)]},
),
(
"Imports of goods in the United States ",
{"entities": [(0, 16, theme),(20, 37, region)]},
),
(
"By township GDP from Agriculture ",
{"entities": [(3, 11, admin),(12, 32, theme)]},
),
(
"In the United States Average hours per day by men spent on Socializing, relaxing, and leisure ",
{"entities": [(3, 20, region),(21, 93, theme)]},
),
(
"By township in U.S. in 1994 Elementary-secondary revenue from local government ",
{"entities": [(3, 11, admin),(15, 19, region),(23, 27, time),(28, 78, theme)]},
),
(
"Helicobacter pylori rate in China ",
{"entities": [(0, 24, theme),(28, 33, region)]},
),
(
"In U.S. in 1985 by state License Taxes of governments ",
{"entities": [(3, 7, region),(11, 15, time),(19, 24, admin),(25, 53, theme)]},
),
(
"In 1978 in South Korea Social Security Rate For Companies by census tract ",
{"entities": [(3, 7, time),(11, 22, region),(23, 57, theme),(61, 73, admin)]},
),
(
"Social Security Rate For Companies in 1967 by province ",
{"entities": [(0, 34, theme),(38, 42, time),(46, 54, admin)]},
),
(
"In Canada in 1969 Mixed Radiological Wastes ",
{"entities": [(3, 9, region),(13, 17, time),(18, 43, theme)]},
),
(
"By census tract CO2 emission (per capita) in US ",
{"entities": [(3, 15, admin),(16, 41, theme),(45, 47, region)]},
),
(
"In 1995 Current charge of governments ",
{"entities": [(3, 7, time),(8, 37, theme)]},
),
(
"In France by township in 2001 Percent of population of Catholic ",
{"entities": [(3, 9, region),(13, 21, admin),(25, 29, time),(30, 63, theme)]},
),
(
"Nonfamily households in 1966 ",
{"entities": [(0, 20, theme),(24, 28, time)]},
),
(
"Special Wastes in 1950 by province in UK ",
{"entities": [(0, 14, theme),(18, 22, time),(26, 34, admin),(38, 40, region)]},
),
(
"By township Population density of American Indian and Alaska Native in China ",
{"entities": [(3, 11, admin),(12, 67, theme),(71, 76, region)]},
),
(
"License plate vanitization rate in China ",
{"entities": [(0, 31, theme),(35, 40, region)]},
),
(
"In 2001 Sales and Gross Receipts Taxes of governments by county ",
{"entities": [(3, 7, time),(8, 53, theme),(57, 63, admin)]},
),
(
"Landfills by state in South Korea ",
{"entities": [(0, 9, theme),(13, 18, admin),(22, 33, region)]},
),
(
"In 2019 by state Number of paid employees ",
{"entities": [(3, 7, time),(11, 16, admin),(17, 41, theme)]},
),
(
"Finance and insurance revenue in 1997 in France ",
{"entities": [(0, 29, theme),(33, 37, time),(41, 47, region)]},
),
(
"In 2018 Estimated annual sales for acc. & tire store by census tract ",
{"entities": [(3, 7, time),(8, 52, theme),(56, 68, admin)]},
),
(
"In China in 1965 by census tract Production workers average for year of Plastics and rubber products manufacturing ",
{"entities": [(3, 8, region),(12, 16, time),(20, 32, admin),(33, 114, theme)]},
),
(
"Percent of population of Under age 18 by county in 1953 in UK ",
{"entities": [(0, 37, theme),(41, 47, admin),(51, 55, time),(59, 61, region)]},
),
(
"Average number of bedrooms of houses by township ",
{"entities": [(0, 36, theme),(40, 48, admin)]},
),
(
"In France Total value of shipments and receipts for services of Seafood product preparation and packaging ",
{"entities": [(3, 9, region),(10, 105, theme)]},
),
(
"By state Family households (families) ",
{"entities": [(3, 8, admin),(9, 37, theme)]},
),
(
"In UK in 2014 by state Average year built ",
{"entities": [(3, 5, region),(9, 13, time),(17, 22, admin),(23, 41, theme)]},
),
(
"By state in 2002 Households with householder living alone ",
{"entities": [(3, 8, admin),(12, 16, time),(17, 57, theme)]},
),
(
"In France Total capital expenditures of Forging and stamping,Cutlery and handtool manufacturing in 2017 ",
{"entities": [(3, 9, region),(10, 95, theme),(99, 103, time)]},
),
(
"In 2007 by township area of Dwarf Scrub (km2) in South Korea ",
{"entities": [(3, 7, time),(11, 19, admin),(20, 45, theme),(49, 60, region)]},
),
(
"Area of Developed, Open Space (km2) in USA ",
{"entities": [(0, 35, theme),(39, 42, region)]},
),
(
"In 1994 Annual payroll of Motor vehicle parts manufacturing in U.S. ",
{"entities": [(3, 7, time),(8, 59, theme),(63, 67, region)]},
),
(
"Elementary-secondary revenue from parent government contributions in 1974 by township in Canada ",
{"entities": [(0, 65, theme),(69, 73, time),(77, 85, admin),(89, 95, region)]},
),
(
"By state average price for honey per pound ",
{"entities": [(3, 8, admin),(9, 42, theme)]},
),
(
"In China by province Estimated annual sales for Sporting goods hobby, musical instrument, & book stores ",
{"entities": [(3, 8, region),(12, 20, admin),(21, 103, theme)]},
),
(
"Estimated land cover types in 1985 ",
{"entities": [(0, 26, theme),(30, 34, time)]},
),
(
"In 1998 in the United States Mercury Effects by census tract ",
{"entities": [(3, 7, time),(11, 28, region),(29, 44, theme),(48, 60, admin)]},
),
(
"Indoor Air Quality in your Home by province in 1985 ",
{"entities": [(0, 31, theme),(35, 43, admin),(47, 51, time)]},
),
(
"In 2001 in the United States Elementary-secondary revenue from other state aid by state ",
{"entities": [(3, 7, time),(11, 28, region),(29, 78, theme),(82, 87, admin)]},
),
(
"Average percent of time engaged in by womenPlaying games in UK in 1983 by township ",
{"entities": [(0, 56, theme),(60, 62, region),(66, 70, time),(74, 82, admin)]},
),
(
"By state Households with one or more people 65 years and over in South Korea ",
{"entities": [(3, 8, admin),(9, 61, theme),(65, 76, region)]},
),
(
"Difference in number of people of people who are confirm to be infected by 2019-Nov Coronavirus by province in 1994 in US ",
{"entities": [(0, 95, theme),(99, 107, admin),(111, 115, time),(119, 121, region)]},
),
(
"Exports value of firms in 1989 by state ",
{"entities": [(0, 22, theme),(26, 30, time),(34, 39, admin)]},
),
(
"In 1976 GDP from Services in U.S. ",
{"entities": [(3, 7, time),(8, 25, theme),(29, 33, region)]},
),
(
"Elementary-secondary revenue from transportation programs by province in 1963 ",
{"entities": [(0, 57, theme),(61, 69, admin),(73, 77, time)]},
),
(
"GDP from Mining in France in 1995 by township ",
{"entities": [(0, 15, theme),(19, 25, region),(29, 33, time),(37, 45, admin)]},
),
(
"Credit Rating by province in UK in 1983 ",
{"entities": [(0, 13, theme),(17, 25, admin),(29, 31, region),(35, 39, time)]},
),
(
"Lead (Pb) by state in 2012 ",
{"entities": [(0, 9, theme),(13, 18, admin),(22, 26, time)]},
),
(
"Annual payroll in 2003 by census tract ",
{"entities": [(0, 14, theme),(18, 22, time),(26, 38, admin)]},
),
(
"In 1972 by state in China License taxes of governments ",
{"entities": [(3, 7, time),(11, 16, admin),(20, 25, region),(26, 54, theme)]},
),
(
"Elementary-secondary revenue from property taxes in the United States in 1953 by county ",
{"entities": [(0, 48, theme),(52, 69, region),(73, 77, time),(81, 87, admin)]},
),
(
"In US in 1955 difference in number of people of who believe climate change ",
{"entities": [(3, 5, region),(9, 13, time),(14, 74, theme)]},
),
(
"In U.S. monthly estimates of the total dollar value of construction work by province ",
{"entities": [(3, 7, region),(8, 72, theme),(76, 84, admin)]},
),
(
"In UK in 1984 homicide rate by census tract ",
{"entities": [(3, 5, region),(9, 13, time),(14, 27, theme),(31, 43, admin)]},
),
(
"In USA by census tract Individual income tax of governments ",
{"entities": [(3, 6, region),(10, 22, admin),(23, 59, theme)]},
),
(
"In U.S. Number of firms ",
{"entities": [(3, 7, region),(8, 23, theme)]},
),
(
"By census tract in USA Average hours per day spent on Travel related to work in 2000 ",
{"entities": [(3, 15, admin),(19, 22, region),(23, 76, theme),(80, 84, time)]},
),
(
"By province Current spending of elementary-secondary expenditure in 1957 ",
{"entities": [(3, 11, admin),(12, 64, theme),(68, 72, time)]},
),
(
"Percent of population of now married, except separated in USA by census tract in 1976 ",
{"entities": [(0, 54, theme),(58, 61, region),(65, 77, admin),(81, 85, time)]},
),
(
"In 1996 Current charge of governments ",
{"entities": [(3, 7, time),(8, 37, theme)]},
),
(
"Percent change of people living in slums in 1984 ",
{"entities": [(0, 40, theme),(44, 48, time)]},
),
(
"In 2001 in France by census tract Household income ",
{"entities": [(3, 7, time),(11, 17, region),(21, 33, admin),(34, 50, theme)]},
),
(
"Area of Lichens (km2) by census tract ",
{"entities": [(0, 21, theme),(25, 37, admin)]},
),
(
"Exports value of firms in South Korea ",
{"entities": [(0, 22, theme),(26, 37, region)]},
),
(
"By province in 1962 Average hours per day by women spent on Vehicle maintenance and repair services in U.S. ",
{"entities": [(3, 11, admin),(15, 19, time),(20, 99, theme),(103, 107, region)]},
),
(
"In 2020 by county Total cost of materials of Medical equipment and supplies manufacturing in the United States ",
{"entities": [(3, 7, time),(11, 17, admin),(18, 89, theme),(93, 110, region)]},
),
(
"By township in South Korea GDP from Manufacturing ",
{"entities": [(3, 11, admin),(15, 26, region),(27, 49, theme)]},
),
(
"Production workers average for year of Apparel manufacturing by township ",
{"entities": [(0, 60, theme),(64, 72, admin)]},
),
(
"In China professional, scientific, and technical services revenue in 2002 by census tract ",
{"entities": [(3, 8, region),(9, 65, theme),(69, 73, time),(77, 89, admin)]},
),
(
"Radiation Effects by province ",
{"entities": [(0, 17, theme),(21, 29, admin)]},
),
(
"In France by township Age of householder ",
{"entities": [(3, 9, region),(13, 21, admin),(22, 40, theme)]},
),
(
"By township in 1995 Sulfur Dioxide (SO2) in U.S. ",
{"entities": [(3, 11, admin),(15, 19, time),(20, 40, theme),(44, 48, region)]},
),
(
"Renter occupied by township in 1951 in U.S. ",
{"entities": [(0, 15, theme),(19, 27, admin),(31, 35, time),(39, 43, region)]},
),
(
"In South Korea difference in number of people of Christian ",
{"entities": [(3, 14, region),(15, 58, theme)]},
),
(
"In 2006 by county life expectancy in USA ",
{"entities": [(3, 7, time),(11, 17, admin),(18, 33, theme),(37, 40, region)]},
),
(
"Average monthly housing cost in USA ",
{"entities": [(0, 28, theme),(32, 35, region)]},
),
(
"Annual payroll in 1987 ",
{"entities": [(0, 14, theme),(18, 22, time)]},
),
(
"In 1988 in USA Water Temperature by province ",
{"entities": [(3, 7, time),(11, 14, region),(15, 32, theme),(36, 44, admin)]},
),
(
"In USA by county area of Palustrine Forested Wetland (km2) in 1979 ",
{"entities": [(3, 6, region),(10, 16, admin),(17, 58, theme),(62, 66, time)]},
),
(
"In Canada by county in 1989 Lead (Pb) ",
{"entities": [(3, 9, region),(13, 19, admin),(23, 27, time),(28, 37, theme)]},
),
(
"In U.S. in 1973 by state helicobacter pylori rate ",
{"entities": [(3, 7, region),(11, 15, time),(19, 24, admin),(25, 49, theme)]},
),
(
"Average number of bedrooms of houses in 1970 ",
{"entities": [(0, 36, theme),(40, 44, time)]},
),
(
"In the United States by county Age of householder ",
{"entities": [(3, 20, region),(24, 30, admin),(31, 49, theme)]},
),
(
"By county in USA in 1992 Estimated annual sales for Family clothing stores ",
{"entities": [(3, 9, admin),(13, 16, region),(20, 24, time),(25, 74, theme)]},
),
(
"Income Taxes of governments in 1965 ",
{"entities": [(0, 27, theme),(31, 35, time)]},
),
(
"Radon by township in 1952 ",
{"entities": [(0, 5, theme),(9, 17, admin),(21, 25, time)]},
),
(
"By township Wages ",
{"entities": [(3, 11, admin),(12, 17, theme)]},
),
(
"In 2001 in UK by census tract Households with male householder, no wife present, family ",
{"entities": [(3, 7, time),(11, 13, region),(17, 29, admin),(30, 87, theme)]},
),
(
"In US Percent change of death among children under 5 due to pediatric cancer ",
{"entities": [(3, 5, region),(6, 76, theme)]},
),
(
"By province Estimated annual sales for Mens clothing stores in UK in 1958 ",
{"entities": [(3, 11, admin),(12, 59, theme),(63, 65, region),(69, 73, time)]},
),
(
"By township in Canada Income Taxes of governments ",
{"entities": [(3, 11, admin),(15, 21, region),(22, 49, theme)]},
),
(
"By county Total capital expenditures of Manufacturing and reproducing magnetic and optical media in USA ",
{"entities": [(3, 9, admin),(10, 96, theme),(100, 103, region)]},
),
(
"By state in 1987 in China area of Woody Wetlands (km2) ",
{"entities": [(3, 8, admin),(12, 16, time),(20, 25, region),(26, 54, theme)]},
),
(
"Outdoor Recreation Value Added as Percent of State GDP by county ",
{"entities": [(0, 54, theme),(58, 64, admin)]},
),
(
"Estimated annual sales for Auto & other motor veh. Dealers in 1992 in UK ",
{"entities": [(0, 58, theme),(62, 66, time),(70, 72, region)]},
),
(
"By county in USA Households with male householder, no wife present, family ",
{"entities": [(3, 9, admin),(13, 16, region),(17, 74, theme)]},
),
(
"Capital outlay of elementary-secondary expenditure by census tract in 1957 in France ",
{"entities": [(0, 50, theme),(54, 66, admin),(70, 74, time),(78, 84, region)]},
),
(
"By census tract Insurance trust expenditure of governments ",
{"entities": [(3, 15, admin),(16, 58, theme)]},
),
(
"In US in 2005 Property Taxes of governments ",
{"entities": [(3, 5, region),(9, 13, time),(14, 43, theme)]},
),
(
"Estimated annual sales for Gasoline stations by state ",
{"entities": [(0, 44, theme),(48, 53, admin)]},
),
(
"Estimated annual sales for Health & personal care stores in 1968 ",
{"entities": [(0, 56, theme),(60, 64, time)]},
),
(
"In South Korea Average percent of time engaged in by womenSocializing, relaxing, and leisure ",
{"entities": [(3, 14, region),(15, 92, theme)]},
),
(
"In 2010 difference in population density of Native Hawaiian and Other Pacific Islander by township ",
{"entities": [(3, 7, time),(8, 86, theme),(90, 98, admin)]},
),
(
"In US by state Radiation Effects in 1997 ",
{"entities": [(3, 5, region),(9, 14, admin),(15, 32, theme),(36, 40, time)]},
),
(
"Area of Perennial Ice/Snow (km2) by province ",
{"entities": [(0, 32, theme),(36, 44, admin)]},
),
(
"By township Average year built in 1951 ",
{"entities": [(3, 11, admin),(12, 30, theme),(34, 38, time)]},
),
(
"In China in 2011 Hazardous/Toxic Air Pollutants ",
{"entities": [(3, 8, region),(12, 16, time),(17, 47, theme)]},
),
(
"Average hours per day by women spent on Caring for nonhousehold adults in US in 2001 ",
{"entities": [(0, 70, theme),(74, 76, region),(80, 84, time)]},
),
(
"By county Average percent of time engaged in by menConsumer goods purchases ",
{"entities": [(3, 9, admin),(10, 75, theme)]},
),
(
"In U.S. Interest on general debt of governments in 2003 by census tract ",
{"entities": [(3, 7, region),(8, 47, theme),(51, 55, time),(59, 71, admin)]},
),
(
"By county area of Dwarf Scrub (km2) in 2014 in Canada ",
{"entities": [(3, 9, admin),(10, 35, theme),(39, 43, time),(47, 53, region)]},
),
(
"In France Average hours per day by women spent on Vehicle maintenance and repair services by census tract ",
{"entities": [(3, 9, region),(10, 89, theme),(93, 105, admin)]},
),
(
"In South Korea by township in 1974 Current spending of elementary-secondary expenditure ",
{"entities": [(3, 14, region),(18, 26, admin),(30, 34, time),(35, 87, theme)]},
),
(
"By township Elementary-secondary revenue from local sources in Canada ",
{"entities": [(3, 11, admin),(12, 59, theme),(63, 69, region)]},
),
(
"Average percent of time engaged in Food and drink preparation in 2012 in USA ",
{"entities": [(0, 61, theme),(65, 69, time),(73, 76, region)]},
),
(
"Percent of population of White by county ",
{"entities": [(0, 30, theme),(34, 40, admin)]},
),
(
"In 1961 difference in population density of Hispanic or Latino Origin ",
{"entities": [(3, 7, time),(8, 69, theme)]},
),
(
"Average hours per day by men spent on Religious and spiritual activities in USA by township in 1992 ",
{"entities": [(0, 72, theme),(76, 79, region),(83, 91, admin),(95, 99, time)]},
),
(
"By province in US General sales of governments in 1956 ",
{"entities": [(3, 11, admin),(15, 17, region),(18, 46, theme),(50, 54, time)]},
),
(
"In 1980 Average hours per day spent on Travel related to household activities ",
{"entities": [(3, 7, time),(8, 77, theme)]},
),
(
"In 1982 Average percent of time engaged in Home maintenance, repair, decoration, and construction (not done by self) in UK ",
{"entities": [(3, 7, time),(8, 116, theme),(120, 122, region)]},
),
(
"Used Oil by county ",
{"entities": [(0, 8, theme),(12, 18, admin)]},
),
(
"By province in 1983 Radiation Effects in Canada ",
{"entities": [(3, 11, admin),(15, 19, time),(20, 37, theme),(41, 47, region)]},
),
(
"In France by state Estimated annual sales for total (excl. motor vehicle & parts & gasoline stations) ",
{"entities": [(3, 9, region),(13, 18, admin),(19, 101, theme)]},
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
),

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
    # test_text_list = ["By state Average percent of time engaged in Vehicles in South Korea in 2004 ",
    # "In 1964 Household income ","By census tract in France in 1972 Household income ",
    # "By township in 1969 Estimated annual sales for Electronics & appliance stores ",
    # "By state Average monthly housing cost as percentage of income ",
    # "By township rate of male in Canada ","Number of firms in the United States ",
    # "In the UK in 1955 number of cell phones per 100 person ",
    # "By county in the UK Average hours per day by women spent on Consumer goods purchases ",
    # "Adult obesity by state in 1973 "]
    
    realTitles = [
        'United States Passport Ownership','Interest in United States Presidential Debates September 29-30 2020',
        '1990 Census Data % of Population 65 and Older','Average Temperature for the US States from July 2015,Choropleth Map',
        'Median Household Income in the United States: 2015','Population per square mile by state',
        'Poverty in the United States Percentage of People in Poverty by State','2017 Poverty Rate in the United States',
        'Figure 2 Percentage of People in Poverty for the United States and Puerto Rico: 2018',
        'Change in Divorce Rates Between 1980 and 1990','Figure 1 Percentage of the People Living in Poverty Areas by State: 2006-2010',
        'Share of High School Students Attending a School with a Sworn Law Enforcement Officer','State Rankings',
        'Estimated Median Household Income 2008 Contiguous United States','Amish Population per state (2010)',
        'Percent of People Below Poverty Level 2004','US Population Density',
        'Figure 1. Percent Change in Resident Population for the 50 States the District of Columbia and Puerto Rico: 1990 to 2000',
        'Population per square mile by state' 
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
