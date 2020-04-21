from __future__ import unicode_literals, print_function

import random
import numpy as np
import os
from spacy.util import minibatch, compounding
import spacy
from pathlib import Path
import plac
import en_core_web_sm

# new entity label
LABEL = "THEME"

# training data
# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting

import pickle
# path = 'C:\\Users\\jiali\\Desktop\\MapElementDetection\code\\Name Entity Recognition\\labledTitles.pkl'
# TRAIN_DATA = pickle.load( open( path, "rb" ) )
nlpTest = en_core_web_sm.load()

TRAIN_DATA = [(
"In 1961 by county Intergovernmental revenue of governments in South Korea ",
{"entities": [(18, 58, LABEL)]},
),
(
"In 1995 Annual payroll by state in the UK ",
{"entities": [(8, 22, LABEL)]},
),
(
"Average hours per day by men spent on Work-related activities by census tract ",
{"entities": [(0, 61, LABEL)]},
),
(
"Number of fire points by census tract ",
{"entities": [(0, 21, LABEL)]},
),
(
"Estimated annual sales for All oth. gen. merch. Stores by county in 1983 in South Korea ",
{"entities": [(0, 54, LABEL)]},
),
(
"Intergovernmental expenditure of governments in 1988 by township ",
{"entities": [(0, 44, LABEL)]},
),
(
"Estimated annual sales for Motor vehicle & parts Dealers in 1951 in the United States ",
{"entities": [(0, 56, LABEL)]},
),
(
"Renter occupied in Canada by census tract in 1985 ",
{"entities": [(0, 15, LABEL)]},
),
(
"By state import and export statistics ",
{"entities": [(9, 37, LABEL)]},
),
(
"In 1976 by township in the UK Estimated annual sales for Building material & garden eq. & supplies dealers ",
{"entities": [(30, 106, LABEL)]},
),
(
"License taxes of governments by county in the United States ",
{"entities": [(0, 28, LABEL)]},
),
(
"In the UK sale amounts of beer by state ",
{"entities": [(10, 30, LABEL)]},
),
(
"In the UK Number of paid employees in 1976 by census tract ",
{"entities": [(10, 34, LABEL)]},
),
(
"Population density of separated in 1987 in China ",
{"entities": [(0, 31, LABEL)]},
),
(
"In 1983 Average household size by township in Canada ",
{"entities": [(8, 30, LABEL)]},
),
(
"Estimated annual sales for Food & beverage stores in 1960 ",
{"entities": [(0, 49, LABEL)]},
),
(
"In 1982 social vulnerability index by county ",
{"entities": [(8, 34, LABEL)]},
),
(
"By county GDP (nominal or ppp) ",
{"entities": [(10, 30, LABEL)]},
),
(
"In France Number of employees of Audio and video equipment manufacturing by county ",
{"entities": [(10, 72, LABEL)]},
),
(
"Total capital expenditures of Clay product and refractory manufacturing by census tract in the United States in 1973 ",
{"entities": [(0, 71, LABEL)]},
),
(
"In 2000 NSF funding for Catalogue in France by township ",
{"entities": [(8, 35, LABEL)]},
),
(
"By state in 1990 Household income in South Korea ",
{"entities": [(17, 33, LABEL)]},
),
(
"In 1985 Gross profit of companies by census tract ",
{"entities": [(8, 33, LABEL)]},
),
(
"In Canada Estimated annual sales for total (excl. motor vehicle & parts & gasoline stations) ",
{"entities": [(10, 92, LABEL)]},
),
(
"In 1988 Direct expenditure of governments ",
{"entities": [(8, 41, LABEL)]},
),
(
"By county in 2013 Annual payroll ",
{"entities": [(18, 32, LABEL)]},
),
(
"In 1953 in the UK Population density of White ",
{"entities": [(18, 45, LABEL)]},
),
(
"In 1952 average age ",
{"entities": [(8, 19, LABEL)]},
),
(
"In France Households with householder living alone by county ",
{"entities": [(10, 50, LABEL)]},
),
(
"In 1979 in Canada Insurance benefits and repayments of governments ",
{"entities": [(18, 66, LABEL)]},
),
(
"License Taxes of governments in Canada in 2003 by state ",
{"entities": [(0, 28, LABEL)]},
),
(
"In France in 1976 Households with male householder, no wife present, family ",
{"entities": [(18, 75, LABEL)]},
),
(
"Estimated annual sales for Clothing & clothing accessories stores by census tract in 1968 ",
{"entities": [(0, 65, LABEL)]},
),
(
"In Canada annual average temperature in 2012 ",
{"entities": [(10, 36, LABEL)]},
),
(
"Percent of farms with female principal operator in France ",
{"entities": [(0, 47, LABEL)]},
),
(
"By census tract Average percent of time engaged in Playing with household children, not sports ",
{"entities": [(16, 94, LABEL)]},
),
(
"In 2014 annual average temperature ",
{"entities": [(8, 34, LABEL)]},
),
(
"In 2004 Annual payroll of Seafood product preparation and packaging in Canada ",
{"entities": [(8, 67, LABEL)]},
),
(
"By state in South Korea number of fire points ",
{"entities": [(24, 45, LABEL)]},
),
(
"In Canada in 1965 number of schools by census tract ",
{"entities": [(18, 35, LABEL)]},
),
(
"By county in 1977 Average hours per day spent on Health-related self care ",
{"entities": [(18, 73, LABEL)]},
),
(
"In the UK Production workers annual hours of Other wood product manufacturing in 1956 by state ",
{"entities": [(10, 77, LABEL)]},
),
(
"In 2014 Exports value of firms by county in the United States ",
{"entities": [(8, 30, LABEL)]},
),
(
"Average hours per day by women spent on Relaxing and thinking by township ",
{"entities": [(0, 61, LABEL)]},
),
(
"In 1977 by township annual average temperature ",
{"entities": [(20, 46, LABEL)]},
),
(
"License Taxes of governments by township in China ",
{"entities": [(0, 28, LABEL)]},
),
(
"In 1968 by county Household income ",
{"entities": [(18, 34, LABEL)]},
),
(
"By township Median household income in France in 1991 ",
{"entities": [(12, 35, LABEL)]},
),
(
"By township in South Korea number of people of White ",
{"entities": [(27, 52, LABEL)]},
),
(
"By census tract in the United States in 1982 well-being index ",
{"entities": [(45, 61, LABEL)]},
),
(
"Average percent of time engaged in Telephone calls (to or from) in Canada in 1991 ",
{"entities": [(0, 63, LABEL)]},
),
(
"Difference in race diversity in the United States ",
{"entities": [(0, 28, LABEL)]},
),
(
"In 1984 Average percent of time engaged in by menCaring for household adults ",
{"entities": [(8, 76, LABEL)]},
),
(
"In 1985 Average hours per day by women spent on Arts and entertainment (other than sports) ",
{"entities": [(8, 90, LABEL)]},
),
(
"In France by census tract Number of paid employees ",
{"entities": [(26, 50, LABEL)]},
),
(
"In 1971 Elementary-secondary revenue from school lunch charges ",
{"entities": [(8, 62, LABEL)]},
),
(
"In China by census tract Average number of bedrooms of houses ",
{"entities": [(25, 61, LABEL)]},
),
(
"NBA player origins (per capita) by county ",
{"entities": [(0, 31, LABEL)]},
),
(
"In 2020 Total value of shipments and receipts for services of Machinery manufacturing ",
{"entities": [(8, 85, LABEL)]},
),
(
"In 1954 Average monthly housing cost ",
{"entities": [(8, 36, LABEL)]},
),
(
"By census tract Estimated annual sales for Furniture stores ",
{"entities": [(16, 59, LABEL)]},
),
(
"In 1971 Family households (families) by county ",
{"entities": [(8, 36, LABEL)]},
),
(
"By county people living in poverty areas ",
{"entities": [(10, 40, LABEL)]},
),
(
"In 2001 Number of paid employees ",
{"entities": [(8, 32, LABEL)]},
),
(
"In France Households with male householder, no wife present, family by state ",
{"entities": [(10, 67, LABEL)]},
),
(
"In 2014 Elementary-secondary revenue from parent government contributions ",
{"entities": [(8, 73, LABEL)]},
),
(
"In the UK in 1960 by census tract Average percent of time engaged in Travel related to personal care] ",
{"entities": [(34, 101, LABEL)]},
),
(
"In France number of fire points by census tract ",
{"entities": [(10, 31, LABEL)]},
),
(
"Sale amounts of beer by census tract ",
{"entities": [(0, 20, LABEL)]},
),
(
"Average year built by county in France ",
{"entities": [(0, 18, LABEL)]},
),
(
"Number of paid employees in China by county in 2006 ",
{"entities": [(0, 24, LABEL)]},
),
(
"By township Estimated annual sales for Mens clothing stores in 1995 in China ",
{"entities": [(12, 59, LABEL)]},
),
(
"By township Average hours per day by women spent on Other income-generating activities ",
{"entities": [(12, 86, LABEL)]},
),
(
"Number of patent in the United States by township ",
{"entities": [(0, 16, LABEL)]},
),
(
"In 1978 in the UK by state Estimated annual sales for Home furnishings stores ",
{"entities": [(27, 77, LABEL)]},
),
(
"By county in 1995 Households with female householder, no husband present, family ",
{"entities": [(18, 80, LABEL)]},
),
(
"In the UK by county number of earthquake ",
{"entities": [(20, 40, LABEL)]},
),
(
"In Canada in 1971 General sales of governments by county ",
{"entities": [(18, 46, LABEL)]},
),
(
"By census tract in 1950 Number of paid employees ",
{"entities": [(24, 48, LABEL)]},
),
(
"In Canada by county Total households ",
{"entities": [(20, 36, LABEL)]},
),
(
"In 1965 Production workers annual hours of Nonmetallic mineral product manufacturing in France by township ",
{"entities": [(8, 84, LABEL)]},
),
(
"In 1993 helicobacter pylori rate ",
{"entities": [(8, 32, LABEL)]},
),
(
"In the United States in 1967 by township Average square footage of houses ",
{"entities": [(41, 73, LABEL)]},
),
(
"General revenue of governments in China ",
{"entities": [(0, 30, LABEL)]},
),
(
"Elementary-secondary revenue from property taxes in 1997 ",
{"entities": [(0, 48, LABEL)]},
),
(
"Married-couple family in China in 1954 by county ",
{"entities": [(0, 21, LABEL)]},
),
(
"In 2004 by township Average hours per day spent on Attending household children events ",
{"entities": [(20, 86, LABEL)]},
),
(
"In 1951 by state General revenue of governments ",
{"entities": [(17, 47, LABEL)]},
),
(
"In 1998 Elementary-secondary revenue from property taxes by township ",
{"entities": [(8, 56, LABEL)]},
),
(
"In China flu incidence by census tract in 1991 ",
{"entities": [(9, 22, LABEL)]},
),
(
"GDP (nominal or ppp) per capita by county in China in 1998 ",
{"entities": [(0, 31, LABEL)]},
),
(
"In 1974 in the United States number of people of Catholic by county ",
{"entities": [(29, 57, LABEL)]},
),
(
"In China in 2003 annual average temperature ",
{"entities": [(17, 43, LABEL)]},
),
(
"In South Korea Number of firms in 1996 by county ",
{"entities": [(15, 30, LABEL)]},
),
(
"In China Family households (families) in 1956 by township ",
{"entities": [(9, 37, LABEL)]},
),
(
"In 2017 Average square footage of houses in South Korea ",
{"entities": [(8, 40, LABEL)]},
),
(
"By township in the United States Percent change of retailers of personal computer ",
{"entities": [(33, 81, LABEL)]},
),
(
"In 1966 Average household size ",
{"entities": [(8, 30, LABEL)]},
),
(
"In China by census tract Average square footage of houses ",
{"entities": [(25, 57, LABEL)]},
),
(
"By census tract Average poverty level for household ",
{"entities": [(16, 51, LABEL)]},
),
(
"By census tract number of academic articles published in 2005 ",
{"entities": [(16, 53, LABEL)]},
),
(
"Estimated annual sales for Pharmacies & drug stores in 2005 ",
{"entities": [(0, 51, LABEL)]},
),
(
"Estimated annual sales for Mens clothing stores by census tract ",
{"entities": [(0, 47, LABEL)]},
),
(
"In the United States in 1985 Households with one or more people 65 years and over ",
{"entities": [(29, 81, LABEL)]},
),
(
"By county in 1969 in Canada Total Taxes of governments ",
{"entities": [(28, 54, LABEL)]},
),
(
"In France by state rate of male ",
{"entities": [(19, 31, LABEL)]},
),
(
"In 1984 in France by township Percent change of people living in slums ",
{"entities": [(30, 70, LABEL)]},
),
(
"In 1964 in France Estimated annual sales for Food & beverage stores ",
{"entities": [(18, 67, LABEL)]},
),
(
"In 1960 in the United States by census tract Annual payroll ",
{"entities": [(45, 59, LABEL)]},
),
(
"By state in 2003 in the UK Population density of Native Hawaiian and Other Pacific Islander ",
{"entities": [(27, 91, LABEL)]},
),
(
"In 1993 in Canada Elementary-secondary revenue from compensatory programs by county ",
{"entities": [(18, 73, LABEL)]},
),
(
"In China Household income in 1995 ",
{"entities": [(9, 25, LABEL)]},
),
(
"By county in 2004 Average percent of time engaged in Purchasing goods and services in South Korea ",
{"entities": [(18, 82, LABEL)]},
),
(
"In 1956 in France Current spending of elementary-secondary expenditure ",
{"entities": [(18, 70, LABEL)]},
),
(
"Percent of planted soybeans by acreage in Canada ",
{"entities": [(0, 38, LABEL)]},
),
(
"Number of Olympic game awards in the UK by county in 1995 ",
{"entities": [(0, 29, LABEL)]},
),
(
"In 1956 Households with one or more people 65 years and over in China ",
{"entities": [(8, 60, LABEL)]},
),
(
"Production workers annual wages of Primary metal manufacturing in South Korea by county ",
{"entities": [(0, 62, LABEL)]},
),
(
"Percent change of frauds in 1983 by state ",
{"entities": [(0, 24, LABEL)]},
),
(
"By state in 1969 in Canada Households with male householder, no wife present, family ",
{"entities": [(27, 84, LABEL)]},
),
(
"In the UK in 2001 Population density of people with a bachelor's degree or higher by township ",
{"entities": [(18, 81, LABEL)]},
),
(
"By state in China lung cancer mortality rate ",
{"entities": [(18, 44, LABEL)]},
),
(
"By county Average monthly housing cost as percentage of income ",
{"entities": [(10, 62, LABEL)]},
),
(
"By census tract in Canada Sales, receipts, or value of shipments of firms in 1999 ",
{"entities": [(26, 73, LABEL)]},
),
(
"By township Insurance trust revenue of governments in 1974 ",
{"entities": [(12, 50, LABEL)]},
),
(
"In 1980 in the UK Number of firms by census tract ",
{"entities": [(18, 33, LABEL)]},
),
(
"Household income by county in 2000 in the United States ",
{"entities": [(0, 16, LABEL)]},
),
(
"In South Korea in 1975 Estimated annual sales for Building material & garden eq. & supplies dealers ",
{"entities": [(23, 99, LABEL)]},
),
(
"In 2015 Production workers annual wages of Beverage manufacturing ",
{"entities": [(8, 65, LABEL)]},
),
(
"By county in 1959 divorce rate ",
{"entities": [(18, 30, LABEL)]},
),
(
"Average age in France ",
{"entities": [(0, 11, LABEL)]},
),
(
"By county annual average temperature in 1964 ",
{"entities": [(10, 36, LABEL)]},
),
(
"In China number of earthquake ",
{"entities": [(9, 29, LABEL)]},
),
(
"Sales, receipts, or value of shipments of firms by census tract in 1954 ",
{"entities": [(0, 47, LABEL)]},
),
(
"In France in 2017 by county Average square footage of houses ",
{"entities": [(28, 60, LABEL)]},
),
(
"In the UK Annual payroll of Soap, cleaning compound, and toilet preparation manufacturing by county ",
{"entities": [(10, 89, LABEL)]},
),
(
"Households with male householder, no wife present, family by state ",
{"entities": [(0, 57, LABEL)]},
),
(
"By census tract Average percent of time engaged in Caring for and helping household adults ",
{"entities": [(16, 90, LABEL)]},
),
(
"In the United States Annual payroll of Basic chemical manufacturing in 1975 ",
{"entities": [(21, 67, LABEL)]},
),
(
"By census tract Average number of bedrooms of houses ",
{"entities": [(16, 52, LABEL)]},
),
(
"Elementary-secondary revenue from transportation programs in China ",
{"entities": [(0, 57, LABEL)]},
),
(
"In 1989 gross domestic income (nominal or ppp) ",
{"entities": [(8, 46, LABEL)]},
),
(
"Average percent of time engaged in by menParticipating in religious practices in 1975 by township in the United States ",
{"entities": [(0, 77, LABEL)]},
),
(
"In France Nonfamily households by county ",
{"entities": [(10, 30, LABEL)]},
),
(
"Elementary-secondary revenue from special education by state in China ",
{"entities": [(0, 51, LABEL)]},
),
(
"Mortality associated with arterial hypertension in Canada by state ",
{"entities": [(0, 47, LABEL)]},
),
(
"In France in 1985 by county Production workers annual hours of Electrical equipment manufacturing ",
{"entities": [(28, 97, LABEL)]},
),
(
"Population density of Jewish in the UK ",
{"entities": [(0, 28, LABEL)]},
),
(
"Exports value of firms by county in the UK ",
{"entities": [(0, 22, LABEL)]},
),
(
"By township Number of paid employees in the United States in 2013 ",
{"entities": [(12, 36, LABEL)]},
),
(
"In the UK Average percent of time engaged in by womenExterior maintenance, repair, and decoration by township in 1965 ",
{"entities": [(10, 97, LABEL)]},
),
(
"In the UK Estimated annual sales for General merchandise stores by state in 1983 ",
{"entities": [(10, 63, LABEL)]},
),
(
"In the United States in 1962 by township number of fire points ",
{"entities": [(41, 62, LABEL)]},
),
(
"By county in the United States Average hours per day spent on Lawn and garden care in 1983 ",
{"entities": [(31, 82, LABEL)]},
),
(
"By census tract in South Korea Exports value of firms in 1987 ",
{"entities": [(31, 53, LABEL)]},
),
(
"In South Korea in 2019 difference in number of people of divorced by township ",
{"entities": [(23, 65, LABEL)]},
),
(
"Production workers annual wages of Other general purpose machinery manufacturing by county in South Korea ",
{"entities": [(0, 80, LABEL)]},
),
(
"In 1950 in Canada General sales of governments by township ",
{"entities": [(18, 46, LABEL)]},
),
(
"Intergovernmental revenue of governments by county in 1957 in the UK ",
{"entities": [(0, 40, LABEL)]},
),
(
"In 1975 Married-couple family ",
{"entities": [(8, 29, LABEL)]},
),
(
"By township Exports value of firms in Canada ",
{"entities": [(12, 34, LABEL)]},
),
(
"In 1962 by township Households with male householder, no wife present, family ",
{"entities": [(20, 77, LABEL)]},
),
(
"In China Production workers annual wages of Leather and allied product manufacturing by census tract ",
{"entities": [(9, 84, LABEL)]},
),
(
"By township difference in number of people of Asian in 1960 ",
{"entities": [(12, 51, LABEL)]},
),
(
"By state Population density of people enrolled in Nursery school, people enrolled in preschool in the UK in 1982 ",
{"entities": [(9, 94, LABEL)]},
),
(
"In the UK median rent price ",
{"entities": [(10, 27, LABEL)]},
),
(
"In 1965 Average monthly housing cost ",
{"entities": [(8, 36, LABEL)]},
),
(
"Average poverty level for household in 1959 ",
{"entities": [(0, 35, LABEL)]},
),
(
"By township in the United States Population density of Jewish ",
{"entities": [(33, 61, LABEL)]},
),
(
"Average poverty level for household in the United States in 1968 ",
{"entities": [(0, 35, LABEL)]},
),
(
"Percent of population of Hispanic or Latino Origin in 1962 ",
{"entities": [(0, 50, LABEL)]},
),
(
"By state Production workers annual wages of Rubber product manufacturing in 1974 in the UK ",
{"entities": [(9, 72, LABEL)]},
),
(
"In 2005 Households with one or more people 65 years and over ",
{"entities": [(8, 60, LABEL)]},
),
(
"By county federal government expenditure (per capita) in 2006 ",
{"entities": [(10, 53, LABEL)]},
),
(
"By state in 2012 Family households (families) ",
{"entities": [(17, 45, LABEL)]},
),
(
"Average number of bedrooms of houses in Canada in 1957 ",
{"entities": [(0, 36, LABEL)]},
),
(
"Elementary-secondary revenue from local sources in the United States ",
{"entities": [(0, 47, LABEL)]},
),
(
"Average poverty level for household in South Korea by state ",
{"entities": [(0, 35, LABEL)]},
),
(
"In France Total cost of materials of Cement and concrete product manufacturing in 1955 ",
{"entities": [(10, 78, LABEL)]},
),
(
"By township in Canada Estimated annual sales for Furniture stores ",
{"entities": [(22, 65, LABEL)]},
),
(
"By county unemployment rate in the United States in 1999 ",
{"entities": [(10, 27, LABEL)]},
),
(
"In 1951 in China by township number of schools ",
{"entities": [(29, 46, LABEL)]},
),
(
"In South Korea Family households (families) ",
{"entities": [(15, 43, LABEL)]},
),
(
"Number of employees of Electrical equipment, appliance, and component manufacturing by township in 1977 ",
{"entities": [(0, 83, LABEL)]},
),
(
"In 1967 in South Korea Estimated annual sales for Nonstore retailers by township ",
{"entities": [(23, 68, LABEL)]},
),
(
"Estimated annual sales for Gasoline stations by state ",
{"entities": [(0, 44, LABEL)]},
),
(
"In 2014 by census tract divorce rate ",
{"entities": [(24, 36, LABEL)]},
),
(
"General sales of governments in South Korea by township ",
{"entities": [(0, 28, LABEL)]},
),
(
"By township Households with one or more people 65 years and over ",
{"entities": [(12, 64, LABEL)]},
),
(
"In Canada Households with one or more people 65 years and over ",
{"entities": [(10, 62, LABEL)]},
),
(
"In 2020 Estimated annual sales for Mens clothing stores by census tract ",
{"entities": [(8, 55, LABEL)]},
),
(
"By census tract Family households (families) ",
{"entities": [(16, 44, LABEL)]},
),
(
"By census tract in the United States in 1990 Elementary-secondary revenue from parent government contributions ",
{"entities": [(45, 110, LABEL)]},
),
(
"In 1989 in Canada by census tract Population density of Muslim ",
{"entities": [(34, 62, LABEL)]},
),
(
"In Canada in 1997 by county Elementary-secondary revenue from local government ",
{"entities": [(28, 78, LABEL)]},
),
(
"In 2015 Elementary-secondary expenditure by county ",
{"entities": [(8, 40, LABEL)]},
),
(
"By census tract in the UK Elementary-secondary revenue from other state aid ",
{"entities": [(26, 75, LABEL)]},
),
(
"In 2008 Estimated annual sales for Building material & garden eq. & supplies dealers in the UK ",
{"entities": [(8, 84, LABEL)]},
),
(
"By township in 1971 Elementary-secondary revenue from general formula assistance in China ",
{"entities": [(20, 80, LABEL)]},
),
(
"By state in South Korea number of fire points in 1956 ",
{"entities": [(24, 45, LABEL)]},
),
(
"In the UK Population density of dentist in 1960 by census tract ",
{"entities": [(10, 39, LABEL)]},
),
(
"In 1976 Household income ",
{"entities": [(8, 24, LABEL)]},
),
(
"By census tract divorce rate in 1954 in the UK ",
{"entities": [(16, 28, LABEL)]},
),
(
"By state annual average precipitation in France ",
{"entities": [(9, 37, LABEL)]},
),
(
"Average percent of time engaged in Computer use for leisure, excluding games in the United States in 1976 by county ",
{"entities": [(0, 76, LABEL)]},
),
(
"License plate vanitization rate by county in 1960 in China ",
{"entities": [(0, 31, LABEL)]},
),
(
"In France in 1982 by census tract Average hours per day by men spent on Indoor and outdoor maintenance, building, and cleanup activities ",
{"entities": [(34, 136, LABEL)]},
),
(
"Number of fire points in South Korea by census tract ",
{"entities": [(0, 21, LABEL)]},
),
(
"Production workers annual wages of Office furniture (including fixtures) manufacturing by state ",
{"entities": [(0, 86, LABEL)]},
),
(
"Percent change of retailers of personal computer in the United States ",
{"entities": [(0, 48, LABEL)]},
),
(
"In Canada in 1957 CO2 emission (per capita) ",
{"entities": [(18, 43, LABEL)]},
),
(
"In 1976 in South Korea Average monthly housing cost as percentage of income by state ",
{"entities": [(23, 75, LABEL)]},
),
(
"In Canada Estimated annual sales for Warehouse clubs & supercenters by county ",
{"entities": [(10, 67, LABEL)]},
),
(
"Elementary-secondary revenue from other state aid by township in Canada ",
{"entities": [(0, 49, LABEL)]},
),
(
"Interest on debt of governments in 1953 in France ",
{"entities": [(0, 31, LABEL)]},
),
(
"By census tract in Canada burglary per 1000 household ",
{"entities": [(26, 53, LABEL)]},
),
(
"In France Elementary-secondary revenue in 1974 ",
{"entities": [(10, 38, LABEL)]},
),
(
"By county Estimated annual sales for Electronics & appliance stores ",
{"entities": [(10, 67, LABEL)]},
),
(
"Average monthly housing cost as percentage of income by township in Canada ",
{"entities": [(0, 52, LABEL)]},
),
(
"By census tract NSF funding for Catalogue ",
{"entities": [(16, 43, LABEL)]},
),
(
"Median rent price in the United States in 1994 by census tract ",
{"entities": [(0, 17, LABEL)]},
),
(
"By county Estimated annual sales for total ",
{"entities": [(10, 42, LABEL)]},
),
(
"By census tract Total cost of materials of Glass and glass product manufacturing in 1963 ",
{"entities": [(16, 80, LABEL)]},
),
(
"Households with householder living alone in South Korea ",
{"entities": [(0, 40, LABEL)]},
),
(
"In 1953 Average percent of time engaged in by menOrganizational, civic, and religious activities by county in South Korea ",
{"entities": [(8, 96, LABEL)]},
),
(
"Number of people of People who is infected by HIV in South Korea ",
{"entities": [(0, 49, LABEL)]},
),
(
"Average family size in the UK ",
{"entities": [(0, 19, LABEL)]},
),
(
"Number of firms by township ",
{"entities": [(0, 15, LABEL)]},
),
(
"GDP (nominal or ppp) per capita in 1967 by county in South Korea ",
{"entities": [(0, 31, LABEL)]},
),
(
"License plate vanitization rate by census tract ",
{"entities": [(0, 31, LABEL)]},
),
(
"By township in France in 1995 Elementary-secondary revenue from local sources ",
{"entities": [(30, 77, LABEL)]},
),
(
"By state in 1981 in South Korea Population density of people enrolled in Elementary school (grades 1-8) ",
{"entities": [(32, 103, LABEL)]},
),
(
"By census tract in France number of earthquake in 2020 ",
{"entities": [(26, 46, LABEL)]},
),
(
"Average percent of time engaged in Other income-generating activities in 1969 in China ",
{"entities": [(0, 69, LABEL)]},
),
(
"By county annual average temperature in 2012 ",
{"entities": [(10, 36, LABEL)]},
),
(
"In the United States by census tract in 1967 Production workers annual hours of Foundries ",
{"entities": [(45, 89, LABEL)]},
),
(
"Average family size in 1998 ",
{"entities": [(0, 19, LABEL)]},
),
(
"By county number of earthquake ",
{"entities": [(10, 30, LABEL)]},
),
(
"Mortality associated with arterial hypertension in 1986 by county in Canada ",
{"entities": [(0, 47, LABEL)]},
),
(
"In the United States Average hours per day by women spent on Helping nonhousehold adults ",
{"entities": [(21, 88, LABEL)]},
),
(
"Average hours per day spent on Travel related to telephone calls by county in France ",
{"entities": [(0, 64, LABEL)]},
),
(
"In South Korea annual average precipitation by township in 1960 ",
{"entities": [(15, 43, LABEL)]},
),
(
"Number of fire points in France ",
{"entities": [(0, 21, LABEL)]},
),
(
"Estimated annual sales for Auto parts in the UK ",
{"entities": [(0, 37, LABEL)]},
),
(
"Total capital expenditures of Computer and electronic product manufacturing in 1994 in Canada by township ",
{"entities": [(0, 75, LABEL)]},
),
(
"Population density of separated in the United States ",
{"entities": [(0, 31, LABEL)]},
),
(
"In 1992 number of earthquake ",
{"entities": [(8, 28, LABEL)]},
),
(
"In Canada by township in 1972 Total value of shipments and receipts for services of Household and institutional furniture and kitchen cabinet manufacturing ",
{"entities": [(30, 155, LABEL)]},
),
(
"Family households (families) by state in Canada in 2004 ",
{"entities": [(0, 28, LABEL)]},
),
(
"By county Liquor stores expenditure of governments ",
{"entities": [(10, 50, LABEL)]},
),
(
"In 2011 by county in the UK annual average precipitation ",
{"entities": [(28, 56, LABEL)]},
),
(
"Average square footage of houses in France in 1966 ",
{"entities": [(0, 32, LABEL)]},
),
(
"In 1972 Capital outlay of elementary-secondary expenditure by state in the United States ",
{"entities": [(8, 58, LABEL)]},
),
(
"In 2016 Average year built by state ",
{"entities": [(8, 26, LABEL)]},
),
(
"In 2019 in Canada percent of houses with annual income of $50,000 and less ",
{"entities": [(18, 74, LABEL)]},
),
(
"In 1967 in South Korea Total value of shipments and receipts for services of Audio and video equipment manufacturing ",
{"entities": [(23, 116, LABEL)]},
),
(
"In 1951 in France by township number of libraries ",
{"entities": [(30, 49, LABEL)]},
),
(
"Number of Olympic game awards in 1959 by county in the UK ",
{"entities": [(0, 29, LABEL)]},
),
(
"In the UK Population density of people who are alumni of OSU by state ",
{"entities": [(10, 60, LABEL)]},
),
(
"Households with one or more people under 18 years by township in 1996 in the UK ",
{"entities": [(0, 49, LABEL)]},
),
(
"By census tract Elementary-secondary revenue from vocational programs ",
{"entities": [(16, 69, LABEL)]},
),
(
"In France by state in 1975 Total revenue of governments ",
{"entities": [(27, 55, LABEL)]},
),
(
"By county in France number of people of retailers of personal computer ",
{"entities": [(20, 70, LABEL)]},
),
(
"By county Estimated annual sales for Womens clothing stores in 1969 ",
{"entities": [(10, 59, LABEL)]},
),
(
"In France in 1956 annual average temperature ",
{"entities": [(18, 44, LABEL)]},
),
(
"By state in the United States number of earthquake in 1960 ",
{"entities": [(30, 50, LABEL)]},
),
(
"Average family size in the UK ",
{"entities": [(0, 19, LABEL)]},
),
(
"Average monthly housing cost as percentage of income by census tract ",
{"entities": [(0, 52, LABEL)]},
),
(
"In the United States in 1978 by township Average percent of time engaged in by menTravel related to work ",
{"entities": [(41, 104, LABEL)]},
),
(
"In South Korea Average percent of time engaged in by womenHelping household children with Homework ",
{"entities": [(15, 98, LABEL)]},
),
(
"Median rent price by county ",
{"entities": [(0, 17, LABEL)]},
),
(
"In France number of people of Under age 18 ",
{"entities": [(10, 42, LABEL)]},
),
(
"In 2018 percent of houses with annual income of $50,000 and less ",
{"entities": [(8, 64, LABEL)]},
),
(
"In 1987 Production workers annual wages of Iron and steel mills and ferroalloy manufacturing by census tract ",
{"entities": [(8, 92, LABEL)]},
),
(
"In 1966 GDP (nominal or ppp) per capita by state in France ",
{"entities": [(8, 39, LABEL)]},
),
(
"In South Korea in 1985 well-being index ",
{"entities": [(23, 39, LABEL)]},
),
(
"By state in 1992 in China Production workers annual wages of Textile furnishings mills ",
{"entities": [(26, 86, LABEL)]},
),
(
"By state in 2009 in Canada Elementary-secondary revenue from local government ",
{"entities": [(27, 77, LABEL)]},
),
(
"Number of employees of Sugar and confectionery product manufacturing in the UK ",
{"entities": [(0, 68, LABEL)]},
),
(
"In Canada Households with one or more people 65 years and over in 1999 ",
{"entities": [(10, 62, LABEL)]},
),
(
"By county in China difference in number of people of women that were screened for breast and cervical cancer by jurisdiction ",
{"entities": [(19, 124, LABEL)]},
),
(
"In 1978 Estimated annual sales for Other general merch. Stores ",
{"entities": [(8, 62, LABEL)]},
),
(
"By county Elementary-secondary revenue from local government in 1973 ",
{"entities": [(10, 60, LABEL)]},
),
(
"By township Estimated annual sales for Home furnishings stores in the UK ",
{"entities": [(12, 62, LABEL)]},
),
(
"Estimated annual sales for All oth. gen. merch. Stores in France in 1965 ",
{"entities": [(0, 54, LABEL)]},
),
(
"In 1955 Number of employees of Other chemical product and preparation manufacturing ",
{"entities": [(8, 83, LABEL)]},
),
(
"Total capital expenditures of Nonmetallic mineral product manufacturing by census tract in 1996 in the UK ",
{"entities": [(0, 71, LABEL)]},
),
(
"Elementary-secondary revenue in Canada ",
{"entities": [(0, 28, LABEL)]},
),
(
"In the UK by county Total capital expenditures of Other wood product manufacturing ",
{"entities": [(20, 82, LABEL)]},
),
(
"In South Korea Production workers annual hours of Architectural and structural metals manufacturing ",
{"entities": [(15, 99, LABEL)]},
),
(
"By county in the United States Annual payroll ",
{"entities": [(31, 45, LABEL)]},
),
(
"In 1961 in South Korea annual average precipitation by census tract ",
{"entities": [(23, 51, LABEL)]},
),
(
"Production workers annual wages of Textile furnishings mills by county in 2012 ",
{"entities": [(0, 60, LABEL)]},
),
(
"Corporate income tax of governments in 2013 ",
{"entities": [(0, 35, LABEL)]},
),
(
"By census tract Average number of bedrooms of houses ",
{"entities": [(16, 52, LABEL)]},
),
(
"Average poverty level for household in France in 2001 by township ",
{"entities": [(0, 35, LABEL)]},
),
(
"Percent change of divorced in 2006 in South Korea by township ",
{"entities": [(0, 26, LABEL)]},
),
(
"By census tract Percent of population of people enrolled in College or graduate school in 1977 ",
{"entities": [(16, 86, LABEL)]},
),
(
"Sales, receipts, or value of shipments of firms by township in 1970 ",
{"entities": [(0, 47, LABEL)]},
),
(
"In China in 1989 Households with male householder, no wife present, family ",
{"entities": [(17, 74, LABEL)]},
),
(
"Annual average precipitation by township in 1984 ",
{"entities": [(0, 28, LABEL)]},
),
(
"Number of McDonald's by census tract ",
{"entities": [(0, 20, LABEL)]},
),
(
"In France in 1989 Production workers annual wages of Leather and allied product manufacturing ",
{"entities": [(18, 93, LABEL)]},
),
(
"By state in France Number of employees of Textile and fabric finishing and fabric coating mills ",
{"entities": [(19, 95, LABEL)]},
),
(
"In China by census tract in 1971 Average percent of time engaged in by womenAttending sporting or recreational events ",
{"entities": [(33, 117, LABEL)]},
),
(
"In 1990 in the United States Taxes of governments ",
{"entities": [(29, 49, LABEL)]},
),
(
"Sales, receipts, or value of shipments of firms in 1974 ",
{"entities": [(0, 47, LABEL)]},
),
(
"Number of paid employees in the United States ",
{"entities": [(0, 24, LABEL)]},
),
(
"Annual average precipitation in 1993 ",
{"entities": [(0, 28, LABEL)]},
),
(
"In 1987 by county Intergovernmental expenditure of governments in the United States ",
{"entities": [(18, 62, LABEL)]},
),
(
"By state Sales and Gross Receipts Taxes of governments in South Korea in 1972 ",
{"entities": [(9, 54, LABEL)]},
),
(
"By township Elementary-secondary revenue from property taxes in the United States in 1969 ",
{"entities": [(12, 60, LABEL)]},
),
(
"In Canada difference in number of people of frauds in 1964 ",
{"entities": [(10, 50, LABEL)]},
),
(
"In the UK in 1958 Race diversity index by township ",
{"entities": [(18, 38, LABEL)]},
),
(
"Total value of shipments and receipts for services of Resin, synthetic rubber, and artificial synthetic fibers and filaments manufacturing in 2009 ",
{"entities": [(0, 138, LABEL)]},
),
(
"In China Intergovernmental expenditure of governments ",
{"entities": [(9, 53, LABEL)]},
),
(
"By census tract Average monthly housing cost as percentage of income in 1960 in France ",
{"entities": [(16, 68, LABEL)]},
),
(
"Elementary-secondary revenue in the UK ",
{"entities": [(0, 28, LABEL)]},
),
(
"Annual average temperature in Canada in 1986 ",
{"entities": [(0, 26, LABEL)]},
),
(
"In 1978 Sales, receipts, or value of shipments of firms in South Korea by county ",
{"entities": [(8, 55, LABEL)]},
),
(
"In the United States Assistance and subsidies of governments in 2001 ",
{"entities": [(21, 60, LABEL)]},
),
(
"In the UK by county Average hours per day by men spent on Vehicles in 1957 ",
{"entities": [(20, 66, LABEL)]},
),
(
"Age of householder by census tract ",
{"entities": [(0, 18, LABEL)]},
),
(
"In Canada by census tract difference in number of people of people who are alumni of OSU ",
{"entities": [(26, 88, LABEL)]},
),
(
"In 1988 Intergovernmental expenditure of governments ",
{"entities": [(8, 52, LABEL)]},
),
(
"In the United States Production workers annual hours of Rubber product manufacturing ",
{"entities": [(21, 84, LABEL)]},
),
(
"Average price for honey per pound in 1981 by census tract ",
{"entities": [(0, 33, LABEL)]},
),
(
"In 1968 in the United States by township annual average precipitation ",
{"entities": [(41, 69, LABEL)]},
),
(
"In China by state in 1997 number of fire points ",
{"entities": [(26, 47, LABEL)]},
),
(
"Estimated annual sales for Grocery stores by census tract in China ",
{"entities": [(0, 41, LABEL)]},
),
(
"In 1985 Production workers annual hours of Nonmetallic mineral product manufacturing ",
{"entities": [(8, 84, LABEL)]},
),
(
"By state in Canada in 2009 number of earthquake ",
{"entities": [(27, 47, LABEL)]},
),
(
"Annual payroll by township in the United States in 1963 ",
{"entities": [(0, 14, LABEL)]},
),
(
"In 1955 Estimated annual sales for Building mat. & sup. dealers in France ",
{"entities": [(8, 63, LABEL)]},
),
(
"Average monthly housing cost as percentage of income in 1950 by state in the United States ",
{"entities": [(0, 52, LABEL)]},
),
(
"By county in 1974 Average number of bedrooms of houses ",
{"entities": [(18, 54, LABEL)]},
),
(
"Annual average temperature in 1952 in the UK ",
{"entities": [(0, 26, LABEL)]},
),
(
"In 1980 Estimated annual sales for Health & personal care stores in France ",
{"entities": [(8, 64, LABEL)]},
),
(
"In 1987 in China Production workers annual hours of Resin, synthetic rubber, and artificial synthetic fibers and filaments manufacturing by county ",
{"entities": [(17, 136, LABEL)]},
),
(
"Median rent price in 1987 by township in South Korea ",
{"entities": [(0, 17, LABEL)]},
),
(
"Annual average precipitation in the UK in 1974 by state ",
{"entities": [(0, 28, LABEL)]},
),
(
"Estimated annual sales for Department stores by township ",
{"entities": [(0, 44, LABEL)]},
),
(
"In 1965 Estimated annual sales for Building mat. & sup. dealers ",
{"entities": [(8, 63, LABEL)]},
),
(
"Households with householder living alone in 1976 ",
{"entities": [(0, 40, LABEL)]},
),
(
"Average household size by township ",
{"entities": [(0, 22, LABEL)]},
),
(
"By county Percent of population of people enrolled in College or graduate school ",
{"entities": [(10, 80, LABEL)]},
),
(
"In France in 1971 number of fire points by census tract ",
{"entities": [(18, 39, LABEL)]},
),
(
"By state Average number of bedrooms of houses in the United States ",
{"entities": [(9, 45, LABEL)]},
),
(
"Total value of shipments and receipts for services of Textile furnishings mills by census tract ",
{"entities": [(0, 79, LABEL)]},
),
(
"Happiness score by census tract ",
{"entities": [(0, 15, LABEL)]},
),
(
"By county Household income in 1950 in the United States ",
{"entities": [(10, 26, LABEL)]},
),
(
"Total cost of materials of Other wood product manufacturing by state in 2008 in the United States ",
{"entities": [(0, 59, LABEL)]},
),
(
"In 1978 by county in South Korea Exports value of firms ",
{"entities": [(33, 55, LABEL)]},
),
(
"By state in South Korea in 1983 Average family size ",
{"entities": [(32, 51, LABEL)]},
),
(
"Annual payroll in the UK by township in 1959 ",
{"entities": [(0, 14, LABEL)]},
),
(
"In South Korea Total households by township ",
{"entities": [(15, 31, LABEL)]},
),
(
"Average number of bedrooms of houses in China by census tract ",
{"entities": [(0, 36, LABEL)]},
),
(
"In 2006 Elementary-secondary expenditure by township ",
{"entities": [(8, 40, LABEL)]},
),
(
"In 2017 in the UK by census tract firearm death rate ",
{"entities": [(34, 52, LABEL)]},
),
(
"In 2009 by state Married-couple family in South Korea ",
{"entities": [(17, 38, LABEL)]},
),
(
"By state in Canada annual average temperature ",
{"entities": [(19, 45, LABEL)]},
),
(
"By county in 1972 number of earthquake in the United States ",
{"entities": [(18, 38, LABEL)]},
),
(
"By county in 2011 availability of safe drinking water in Canada ",
{"entities": [(18, 53, LABEL)]},
),
(
"Average number of bedrooms of houses in the United States by township ",
{"entities": [(0, 36, LABEL)]},
),
(
"Import and export statistics in 1959 ",
{"entities": [(0, 28, LABEL)]},
),
(
"By census tract Average percent of time engaged in Watching TV ",
{"entities": [(16, 62, LABEL)]},
),
(
"In 2012 Average poverty level for household by census tract ",
{"entities": [(8, 43, LABEL)]},
),
(
"In South Korea Salaries and wages of governments ",
{"entities": [(15, 48, LABEL)]},
),
(
"In the UK percent of households above $200k by county in 1972 ",
{"entities": [(10, 43, LABEL)]},
),
(
"Elementary-secondary revenue from vocational programs by state ",
{"entities": [(0, 53, LABEL)]},
),
(
"Average hours per day spent on Household services in Canada in 1953 ",
{"entities": [(0, 49, LABEL)]},
),
(
"In 1970 in France Family households with own children of the householder under 18 years ",
{"entities": [(18, 87, LABEL)]},
),
(
"Production workers annual hours of Clay product and refractory manufacturing by census tract in 1977 ",
{"entities": [(0, 76, LABEL)]},
),
(
"In France by township Household income ",
{"entities": [(22, 38, LABEL)]},
),
(
"Direct expenditure of governments in 2019 ",
{"entities": [(0, 33, LABEL)]},
),
(
"In 1969 in the United States Elementary-secondary revenue from local government ",
{"entities": [(29, 79, LABEL)]},
),
(
"Exports value of firms in the United States ",
{"entities": [(0, 22, LABEL)]},
),
(
"In Canada Estimated annual sales for Furniture & home furn. Stores by state ",
{"entities": [(10, 66, LABEL)]},
),
(
"Elementary-secondary revenue from compensatory programs in 1976 ",
{"entities": [(0, 55, LABEL)]},
),
(
"By census tract Average hours per day by men spent on Interior cleaning ",
{"entities": [(16, 71, LABEL)]},
),
(
"Total cost of materials of Paper manufacturing by township ",
{"entities": [(0, 46, LABEL)]},
),
(
"Married-couple family in the UK ",
{"entities": [(0, 21, LABEL)]},
),
(
"In the United States by census tract in 1987 Average percent of time engaged in Telephone calls (to or from) ",
{"entities": [(45, 108, LABEL)]},
),
(
"Elementary-secondary expenditure in 1982 in the United States ",
{"entities": [(0, 32, LABEL)]},
),
(
"Renter occupied in France by census tract ",
{"entities": [(0, 15, LABEL)]},
),
(
"In the UK Average monthly housing cost as percentage of income in 1981 ",
{"entities": [(10, 62, LABEL)]},
),
(
"By state Total cost of materials of Other furniture related product manufacturing ",
{"entities": [(9, 81, LABEL)]},
),
(
"In 1951 Household income by township ",
{"entities": [(8, 24, LABEL)]},
),
(
"In 1964 difference in number of people of People whose native language is Russian ",
{"entities": [(8, 81, LABEL)]},
),
(
"Estimated annual sales for Pharmacies & drug stores in 2010 in Canada ",
{"entities": [(0, 51, LABEL)]},
),
(
"Rate of male in the United States ",
{"entities": [(0, 12, LABEL)]},
),
(
"By state Production workers annual hours of Communications equipment manufacturing ",
{"entities": [(9, 82, LABEL)]},
),
(
"In the United States by state in 1999 Production workers annual hours of Sawmills and wood preservation ",
{"entities": [(38, 103, LABEL)]},
),
(
"By state in 1985 Median household income ",
{"entities": [(17, 40, LABEL)]},
),
(
"By census tract in the UK Elementary-secondary revenue from compensatory programs in 2004 ",
{"entities": [(26, 81, LABEL)]},
),
(
"By county Utility expenditure of governments in the UK ",
{"entities": [(10, 44, LABEL)]},
),
(
"Miscellaneous general revenue of governments in the United States ",
{"entities": [(0, 44, LABEL)]},
),
(
"Average percent of time engaged in Storing interior household items, including food in the United States in 2020 ",
{"entities": [(0, 83, LABEL)]},
),
(
"Percent change of who believe climate change in 1979 ",
{"entities": [(0, 44, LABEL)]},
),
(
"Average percent of time engaged in by menCaring for and helping nonhousehold adults in China ",
{"entities": [(0, 83, LABEL)]},
),
(
"By census tract Family households with own children of the householder under 18 years ",
{"entities": [(16, 85, LABEL)]},
),
(
"In 2003 by county Percent of population of above age 65 in Canada ",
{"entities": [(18, 55, LABEL)]},
),
(
"In South Korea Average monthly housing cost ",
{"entities": [(15, 43, LABEL)]},
),
(
"Food insecurity rate in China ",
{"entities": [(0, 20, LABEL)]},
),
(
"In 1970 by township in China Production workers annual wages of Electric lighting equipment manufacturing ",
{"entities": [(29, 105, LABEL)]},
),
(
"Number of employees of Commercial and service industry machinery manufacturing by census tract ",
{"entities": [(0, 78, LABEL)]},
),
(
"By state in 1960 Sales, receipts, or value of shipments of firms ",
{"entities": [(17, 64, LABEL)]},
),
(
"By township Average year built ",
{"entities": [(12, 30, LABEL)]},
),
(
"Average family size in 1989 ",
{"entities": [(0, 19, LABEL)]},
),
(
"In 1975 Nonfamily households ",
{"entities": [(8, 28, LABEL)]},
),
(
"In France in 2006 Average percent of time engaged in by menPlaying with household children, not sports ",
{"entities": [(18, 102, LABEL)]},
),
(
"In 1983 by census tract number of earthquake ",
{"entities": [(24, 44, LABEL)]},
),
(
"In 1963 number of earthquake ",
{"entities": [(8, 28, LABEL)]},
),
(
"Married-couple family by township in France in 2019 ",
{"entities": [(0, 21, LABEL)]},
),
(
"Average percent of time engaged in by menHousehold and personal e-mail and messages by county ",
{"entities": [(0, 83, LABEL)]},
),
(
"In the United States by state Annual payroll in 1969 ",
{"entities": [(30, 44, LABEL)]},
),
(
"In the United States Households with householder living alone ",
{"entities": [(21, 61, LABEL)]},
),
(
"In France in 1951 number of multi-racial households ",
{"entities": [(18, 51, LABEL)]},
),
(
"Average percent of time engaged in by womenAttending meetings, conferences, and training in Canada ",
{"entities": [(0, 88, LABEL)]},
),
(
"In South Korea Current spending of elementary-secondary expenditure in 1957 ",
{"entities": [(15, 67, LABEL)]},
),
(
"In 1954 in Canada Number of employees of Other nonmetallic mineral product manufacturing by state ",
{"entities": [(18, 88, LABEL)]},
),
(
"Number of schools in the UK in 2014 ",
{"entities": [(0, 17, LABEL)]},
),
(
"In the UK Elementary-secondary revenue from state sources by county in 1969 ",
{"entities": [(10, 57, LABEL)]},
),
(
"By state Total capital expenditures of Animal slaughtering and processing ",
{"entities": [(9, 73, LABEL)]},
),
(
"Selective sales of governments in the UK ",
{"entities": [(0, 30, LABEL)]},
),
(
"By census tract in 2019 Average monthly housing cost as percentage of income in France ",
{"entities": [(24, 76, LABEL)]},
),
(
"In 1972 Estimated annual sales for Gasoline stations in France ",
{"entities": [(8, 52, LABEL)]},
),
(
"In 1951 by township in the UK Average percent of time engaged in Arts and entertainment (other than sports) ",
{"entities": [(30, 107, LABEL)]},
),
(
"Cash and security holdings of governments by township in 1993 in France ",
{"entities": [(0, 41, LABEL)]},
),
(
"Production workers annual hours of Nonmetallic mineral product manufacturing by township ",
{"entities": [(0, 76, LABEL)]},
),
(
"By county number of McDonald's ",
{"entities": [(10, 30, LABEL)]},
),
(
"In Canada in 1951 by census tract Exports value of firms ",
{"entities": [(34, 56, LABEL)]},
),
(
"Sales, receipts, or value of shipments of firms by census tract in France ",
{"entities": [(0, 47, LABEL)]},
),
(
"In the UK by county in 1979 Liquor stores revenue of governments ",
{"entities": [(28, 64, LABEL)]},
),
(
"In Canada Number of firms by state ",
{"entities": [(10, 25, LABEL)]},
),
(
"Number of paid employees in South Korea in 1996 by township ",
{"entities": [(0, 24, LABEL)]},
),
(
"Average percent of time engaged in Caring for household adults in 1990 ",
{"entities": [(0, 62, LABEL)]},
),
(
"In the UK Total Taxes of governments ",
{"entities": [(10, 36, LABEL)]},
),
(
"In 2008 Elementary-secondary expenditure by county in Canada ",
{"entities": [(8, 40, LABEL)]},
),
(
"By state in 2003 Number of paid employees ",
{"entities": [(17, 41, LABEL)]},
),
(
"In the UK Percent of planted soybeans by acreage ",
{"entities": [(10, 48, LABEL)]},
),
(
"In France in 1964 by county Estimated annual sales for Gasoline stations ",
{"entities": [(28, 72, LABEL)]},
),
(
"By census tract number of Olympic game awards in 1971 in South Korea ",
{"entities": [(16, 45, LABEL)]},
),
(
"By census tract Direct expenditure of governments ",
{"entities": [(16, 49, LABEL)]},
),
(
"In France in 1987 by county Elementary-secondary revenue from transportation programs ",
{"entities": [(28, 85, LABEL)]},
),
(
"By county Average hours per day by women spent on Household services in 1993 ",
{"entities": [(10, 68, LABEL)]},
),
(
"By township Elementary-secondary revenue from federal sources in 1999 in the UK ",
{"entities": [(12, 61, LABEL)]},
),
(
"In 2007 in the United States Race diversity index by township ",
{"entities": [(29, 49, LABEL)]},
),
(
"By county Nonfamily households ",
{"entities": [(10, 30, LABEL)]},
),
(
"In 2007 by state Total capital expenditures of Hardware manufacturing,Spring and wire product manufacturing in China ",
{"entities": [(17, 107, LABEL)]},
),
(
"In 1952 median rent price in Canada ",
{"entities": [(8, 25, LABEL)]},
),
(
"By township in South Korea Current spending of elementary-secondary expenditure ",
{"entities": [(27, 79, LABEL)]},
),
(
"In 1969 Current spending of elementary-secondary expenditure ",
{"entities": [(8, 60, LABEL)]},
),
(
"In France Estimated annual sales for Elect. shopping & m/o houses ",
{"entities": [(10, 65, LABEL)]},
),
(
"By township in 1975 Total households in Canada ",
{"entities": [(20, 36, LABEL)]},
),
(
"In Canada number of multi-racial households ",
{"entities": [(10, 43, LABEL)]},
),
(
"Number of paid employees by township in Canada ",
{"entities": [(0, 24, LABEL)]},
),
(
"Households with one or more people under 18 years in the United States ",
{"entities": [(0, 49, LABEL)]},
),
(
"In the United States Exports value of firms ",
{"entities": [(21, 43, LABEL)]},
),
(
"In France in 1966 by county Elementary-secondary revenue from general formula assistance ",
{"entities": [(28, 88, LABEL)]},
),
(
"By census tract in 1977 poverty rate ",
{"entities": [(24, 36, LABEL)]},
),
(
"Selective sales of governments in the United States ",
{"entities": [(0, 30, LABEL)]},
),
(
"By census tract Total cost of materials of Household and institutional furniture and kitchen cabinet manufacturing ",
{"entities": [(16, 114, LABEL)]},
),
(
"Production workers average for year of Electric lighting equipment manufacturing in the United States ",
{"entities": [(0, 80, LABEL)]},
),
(
"Number of people of frauds by census tract in 1989 ",
{"entities": [(0, 26, LABEL)]},
),
(
"Percent of population of dentist in 1970 by township ",
{"entities": [(0, 32, LABEL)]},
),
(
"By county in Canada in 1997 Households with one or more people 65 years and over ",
{"entities": [(28, 80, LABEL)]},
),
(
"In China Population density of people whose permanent teeth have been removed because of tooth decay or gum disease ",
{"entities": [(9, 115, LABEL)]},
),
(
"Elementary-secondary revenue from special education in 1996 by state in South Korea ",
{"entities": [(0, 51, LABEL)]},
),
(
"Elementary-secondary revenue from vocational programs by township ",
{"entities": [(0, 53, LABEL)]},
),
(
"By township Family households with own children of the householder under 18 years in 2019 in the United States ",
{"entities": [(12, 81, LABEL)]},
),
(
"By state poverty rate ",
{"entities": [(9, 21, LABEL)]},
),
(
"In Canada Estimated annual sales for total by state ",
{"entities": [(10, 42, LABEL)]},
),
(
"Number of firms in 1973 by state ",
{"entities": [(0, 15, LABEL)]},
),
(
"Intergovernmental revenue of governments by township ",
{"entities": [(0, 40, LABEL)]},
),
(
"Estimated annual sales for Pharmacies & drug stores in the United States ",
{"entities": [(0, 51, LABEL)]},
),
(
"Production workers annual hours of Communications equipment manufacturing in the UK ",
{"entities": [(0, 73, LABEL)]},
),
(
"Household income by census tract in the UK ",
{"entities": [(0, 16, LABEL)]},
),
(
"In China by state Estimated annual sales for Nonstore retailers ",
{"entities": [(18, 63, LABEL)]},
),
(
"By county in 2011 Average hours per day spent on Computer use for leisure, excluding games in the United States ",
{"entities": [(18, 90, LABEL)]},
),
(
"In the United States difference in number of people of people enrolled in High school (grades 9-12) by census tract in 1990 ",
{"entities": [(21, 99, LABEL)]},
),
(
"By state availability of safe drinking water in South Korea ",
{"entities": [(9, 44, LABEL)]},
),
(
"By state federal government expenditure (per capita) in Canada in 1992 ",
{"entities": [(9, 52, LABEL)]},
),
(
"By state in 1986 Median household income ",
{"entities": [(17, 40, LABEL)]},
),
(
"In France in 1964 Average monthly housing cost as percentage of income ",
{"entities": [(18, 70, LABEL)]},
),
(
"Miscellaneous general revenue of governments in 2018 by county in France ",
{"entities": [(0, 44, LABEL)]},
),
(
"In 1976 import and export statistics in the United States ",
{"entities": [(8, 36, LABEL)]},
),
(
"Difference in population density of people who are confirm to be infected by 2019-Nov Coronavirus by state ",
{"entities": [(0, 97, LABEL)]},
),
(
"Number of employees of Other leather and allied product manufacturing in China ",
{"entities": [(0, 69, LABEL)]},
),
(
"By township in 1951 in China Total value of shipments and receipts for services of Other furniture related product manufacturing ",
{"entities": [(29, 128, LABEL)]},
),
(
"In Canada Average percent of time engaged in by menHousehold and personal messages in 2000 ",
{"entities": [(10, 82, LABEL)]},
),
(
"Annual payroll by census tract in China ",
{"entities": [(0, 14, LABEL)]},
),
(
"In 1962 by county Number of firms ",
{"entities": [(18, 33, LABEL)]},
),
(
"By census tract in 2009 Direct expenditure of governments ",
{"entities": [(24, 57, LABEL)]},
),
(
"By township in France mortality associated with arterial hypertension in 1971 ",
{"entities": [(22, 69, LABEL)]},
),
(
"In France Income Taxes of governments in 1995 ",
{"entities": [(10, 37, LABEL)]},
),
(
"Annual average precipitation by census tract ",
{"entities": [(0, 28, LABEL)]},
),
(
"By township in the United States Direct expenditure of governments in 1989 ",
{"entities": [(33, 66, LABEL)]},
),
(
"In France in 1988 adult obesity by census tract ",
{"entities": [(18, 31, LABEL)]},
),
(
"Total capital expenditures of Beverage manufacturing in 2014 in Canada ",
{"entities": [(0, 52, LABEL)]},
),
(
"In 2001 Elementary-secondary revenue from vocational programs by state in the UK ",
{"entities": [(8, 61, LABEL)]},
),
(
"In 1974 in the UK Total expenditure of governments ",
{"entities": [(18, 50, LABEL)]},
),
(
"In the United States Insurance trust expenditure of governments by census tract in 2013 ",
{"entities": [(21, 63, LABEL)]},
),
(
"By state Average monthly housing cost ",
{"entities": [(9, 37, LABEL)]},
),
(
"By county Number of paid employees ",
{"entities": [(10, 34, LABEL)]},
),
(
"In South Korea Nonfamily households by census tract ",
{"entities": [(15, 35, LABEL)]},
),
(
"In 2017 in the UK by county gun violence rate ",
{"entities": [(28, 45, LABEL)]},
),
(
"In 2015 by township General expenditure of governments ",
{"entities": [(20, 54, LABEL)]},
),
(
"In Canada Average percent of time engaged in by menCaring for nonhousehold adults in 2001 by state ",
{"entities": [(10, 81, LABEL)]},
),
(
"In Canada by state Estimated annual sales for Auto parts ",
{"entities": [(19, 56, LABEL)]},
),
(
"Estimated annual sales for acc. & tire store in South Korea by county in 2007 ",
{"entities": [(0, 44, LABEL)]},
),
(
"In Canada Elementary-secondary revenue from general formula assistance by township in 2009 ",
{"entities": [(10, 70, LABEL)]},
),
(
"Nonfamily households in 2016 ",
{"entities": [(0, 20, LABEL)]},
),
(
"By state in South Korea Households with one or more people under 18 years in 2005 ",
{"entities": [(24, 73, LABEL)]},
),
(
"By state in 2006 in the UK lung cancer mortality rate ",
{"entities": [(27, 53, LABEL)]},
),
(
"Difference in number of people of People whose native language is Russian in 1976 ",
{"entities": [(0, 73, LABEL)]},
),
(
"In 1953 Average square footage of houses ",
{"entities": [(8, 40, LABEL)]},
),
(
"By county in the UK number of people of males 15 years and over ",
{"entities": [(20, 63, LABEL)]},
),
(
"In China Average household size ",
{"entities": [(9, 31, LABEL)]},
),
(
"In 1953 by county in Canada number of libraries ",
{"entities": [(28, 47, LABEL)]},
),
(
"In China Annual payroll of Converted paper product manufacturing ",
{"entities": [(9, 64, LABEL)]},
),
(
"In 1953 annual average precipitation ",
{"entities": [(8, 36, LABEL)]},
),
(
"In China Estimated annual sales for acc. & tire store by census tract ",
{"entities": [(9, 53, LABEL)]},
),
(
"Number of earthquake in France by state in 1967 ",
{"entities": [(0, 20, LABEL)]},
),
(
"Elementary-secondary revenue from parent government contributions in South Korea ",
{"entities": [(0, 65, LABEL)]},
),
(
"In the United States Households with female householder, no husband present, family by township in 2014 ",
{"entities": [(21, 83, LABEL)]},
),
(
"Elementary-secondary revenue from state sources in the UK ",
{"entities": [(0, 47, LABEL)]},
),
(
"In 1990 in France number of earthquake by township ",
{"entities": [(18, 38, LABEL)]},
),
(
"In 2013 Estimated annual sales for total (excl. gasoline stations) ",
{"entities": [(8, 66, LABEL)]},
),
(
"Sales, receipts, or value of shipments of firms by census tract ",
{"entities": [(0, 47, LABEL)]},
),
(
"By township annual average temperature in 2001 in South Korea ",
{"entities": [(12, 38, LABEL)]},
),
(
"Price of land in the United States ",
{"entities": [(0, 13, LABEL)]},
),
(
"Average percent of time engaged in Computer use for leisure, excluding games by township in 2014 ",
{"entities": [(0, 76, LABEL)]},
),
(
"In 1956 in France Elementary-secondary revenue from compensatory programs ",
{"entities": [(18, 73, LABEL)]},
),
(
"Married-couple family in 2018 ",
{"entities": [(0, 21, LABEL)]},
),
(
"In 2017 in the United States Estimated annual sales for Food services & drinking places ",
{"entities": [(29, 87, LABEL)]},
),
(
"In 1993 by census tract Households with female householder, no husband present, family ",
{"entities": [(24, 86, LABEL)]},
),
(
"By township Average percent of time engaged in by menVolunteering (organizational and civic activities) ",
{"entities": [(12, 103, LABEL)]},
),
(
"In 1962 in France Average monthly housing cost by county ",
{"entities": [(18, 46, LABEL)]},
),
(
"Social vulnerability index by county ",
{"entities": [(0, 26, LABEL)]},
),
(
"In South Korea Percent of population of per Walmart store ",
{"entities": [(15, 57, LABEL)]},
),
(
"By state number of fire points ",
{"entities": [(9, 30, LABEL)]},
),
(
"In 2014 annual average precipitation by township ",
{"entities": [(8, 36, LABEL)]},
),
(
"Production workers annual wages of Industrial machinery manufacturing by county in 2009 in China ",
{"entities": [(0, 69, LABEL)]},
),
(
"Exports value of firms in 1980 in the UK ",
{"entities": [(0, 22, LABEL)]},
),
(
"In 1986 Annual payroll ",
{"entities": [(8, 22, LABEL)]},
),
(
"Number of earthquake in France in 1973 ",
{"entities": [(0, 20, LABEL)]},
),
(
"In 1957 by township Household income in Canada ",
{"entities": [(20, 36, LABEL)]},
),
(
"Age of householder in 1959 by township ",
{"entities": [(0, 18, LABEL)]},
),
(
"Elementary-secondary revenue from other state aid by census tract ",
{"entities": [(0, 49, LABEL)]},
),
(
"By state in South Korea average price for honey per pound in 1976 ",
{"entities": [(24, 57, LABEL)]},
),
(
"In 1956 by state Average percent of time engaged in Religious and spiritual activities ",
{"entities": [(17, 86, LABEL)]},
),
(
"Total value of shipments and receipts for services of Pulp, paper, and paperboard mills in 1990 in South Korea by state ",
{"entities": [(0, 87, LABEL)]},
),
(
"Mortality associated with arterial hypertension in 2006 in the UK ",
{"entities": [(0, 47, LABEL)]},
),
(
"In 2020 by township in China Income Taxes of governments ",
{"entities": [(29, 56, LABEL)]},
),
(
"In Canada Estimated annual sales for All oth. gen. merch. Stores ",
{"entities": [(10, 64, LABEL)]},
),
(
"Elementary-secondary revenue in South Korea ",
{"entities": [(0, 28, LABEL)]},
),
(
"Average monthly housing cost in China ",
{"entities": [(0, 28, LABEL)]},
),
(
"By county Average hours per day by women spent on Lawn and garden care in the UK ",
{"entities": [(10, 70, LABEL)]},
),
(
"Households with female householder, no husband present, family in 1995 by township in France ",
{"entities": [(0, 62, LABEL)]},
),
(
"By township in 1997 freedom index ",
{"entities": [(20, 33, LABEL)]},
),
(
"In 2013 by state Estimated annual sales for Building material & garden eq. & supplies dealers in Canada ",
{"entities": [(17, 93, LABEL)]},
),
(
"In 1982 in France Number of firms ",
{"entities": [(18, 33, LABEL)]},
),
(
"Percent of forest area by state in France ",
{"entities": [(0, 22, LABEL)]},
),
(
"In China in 1968 by census tract Average monthly housing cost ",
{"entities": [(33, 61, LABEL)]},
),
(
"In 1963 Direct expenditure of governments in Canada by census tract ",
{"entities": [(8, 41, LABEL)]},
),
(
"Adult obesity in 2005 ",
{"entities": [(0, 13, LABEL)]},
),
(
"Average monthly housing cost as percentage of income in 1993 by state ",
{"entities": [(0, 52, LABEL)]},
),
(
"Production workers annual hours of Household and institutional furniture and kitchen cabinet manufacturing by township in 1968 in France ",
{"entities": [(0, 106, LABEL)]},
),
(
"Number of people of White by census tract ",
{"entities": [(0, 25, LABEL)]},
),
(
"In 1963 Average percent of time engaged in by womenInterior maintenance, repair, and decoration by township in China ",
{"entities": [(8, 95, LABEL)]},
),
(
"In Canada economic growth rate ",
{"entities": [(10, 30, LABEL)]},
),
(
"In 1996 in the United States Household income ",
{"entities": [(29, 45, LABEL)]},
),
(
"In France in 1956 number of people of people below poverty level by census tract ",
{"entities": [(18, 64, LABEL)]},
),
(
"Difference in race diversity in 1967 by census tract in China ",
{"entities": [(0, 28, LABEL)]},
),
(
"Annual average temperature in 2007 by census tract ",
{"entities": [(0, 26, LABEL)]},
),
(
"In 1992 Age of householder ",
{"entities": [(8, 26, LABEL)]},
),
(
"By county Elementary-secondary revenue from property taxes ",
{"entities": [(10, 58, LABEL)]},
),
(
"In China by state in 2011 number of fixed residential broadband providers ",
{"entities": [(26, 73, LABEL)]},
),
(
"Percent of households above $200k in South Korea in 1987 ",
{"entities": [(0, 33, LABEL)]},
),
(
"In the UK difference in population density of per Walmart store ",
{"entities": [(10, 63, LABEL)]},
),
(
"In 1972 by county adult obesity ",
{"entities": [(18, 31, LABEL)]},
),
(
"In France by state Estimated annual sales for total (excl. gasoline stations) ",
{"entities": [(19, 77, LABEL)]},
),
(
"Interest on general debt of governments in 1960 in the UK ",
{"entities": [(0, 39, LABEL)]},
),
(
"By census tract in China Total households ",
{"entities": [(25, 41, LABEL)]},
),
(
"By state Percent change of All Race in 2002 in France ",
{"entities": [(9, 35, LABEL)]},
),
(
"In China by county annual average temperature ",
{"entities": [(19, 45, LABEL)]},
),
(
"By county in France in 2000 Estimated annual sales for total (excl. motor vehicle & parts & gasoline stations) ",
{"entities": [(28, 110, LABEL)]},
),
(
"Average hours per day by men spent on Vehicles in the United States by state in 1975 ",
{"entities": [(0, 46, LABEL)]},
),
(
"By state in France Average hours per day by men spent on Storing interior household items, including food in 1972 ",
{"entities": [(19, 105, LABEL)]},
),
(
"Average percent of time engaged in by menOther income-generating activities in 1981 ",
{"entities": [(0, 75, LABEL)]},
),
(
"Households with one or more people 65 years and over by census tract in the United States in 1966 ",
{"entities": [(0, 52, LABEL)]},
),
(
"Median household income in France ",
{"entities": [(0, 23, LABEL)]},
),
(
"By county in Canada Production workers annual wages of Computer and peripheral equipment manufacturing ",
{"entities": [(20, 102, LABEL)]},
),
(
"Estimated annual sales for Motor vehicle & parts Dealers by township in 1974 ",
{"entities": [(0, 56, LABEL)]},
),
(
"In the United States in 1993 happiness score ",
{"entities": [(29, 44, LABEL)]},
),
(
"In 2014 in China Number of employees of Dairy product manufacturing ",
{"entities": [(17, 67, LABEL)]},
),
(
"By census tract number of people of people living in slums in 1985 ",
{"entities": [(16, 58, LABEL)]},
),
(
"By census tract in Canada Annual payroll of Communications equipment manufacturing ",
{"entities": [(26, 82, LABEL)]},
),
(
"By state in 1969 number of earthquake ",
{"entities": [(17, 37, LABEL)]},
),
(
"In South Korea in 1955 by census tract Estimated annual sales for Miscellaneous store retailer ",
{"entities": [(39, 94, LABEL)]},
),
(
"Production workers annual hours of Other textile product mills in 1982 in France by census tract ",
{"entities": [(0, 62, LABEL)]},
),
(
"In 1984 Average square footage of houses in the UK ",
{"entities": [(8, 40, LABEL)]},
),
(
"In the United States by state Estimated annual sales for Warehouse clubs & supercenters in 1989 ",
{"entities": [(30, 87, LABEL)]},
),
(
"Estimated annual sales for Warehouse clubs & supercenters in the UK ",
{"entities": [(0, 57, LABEL)]},
),
(
"By census tract Elementary-secondary revenue from parent government contributions ",
{"entities": [(16, 81, LABEL)]},
),
(
"By township in South Korea in 1976 Average monthly housing cost as percentage of income ",
{"entities": [(35, 87, LABEL)]},
),
(
"In the United States by county in 1982 number of fire points ",
{"entities": [(39, 60, LABEL)]},
),
(
"In China Liquor stores revenue of governments by township ",
{"entities": [(9, 45, LABEL)]},
),
(
"By township in Canada Elementary-secondary revenue from property taxes in 1973 ",
{"entities": [(22, 70, LABEL)]},
),
(
"In 1973 Elementary-secondary revenue from local government by county ",
{"entities": [(8, 58, LABEL)]},
),
(
"General sales of governments in 1998 ",
{"entities": [(0, 28, LABEL)]},
),
(
"In 2016 by state in Canada Average hours per day by women spent on Care for animals and pets, not veterinary care ",
{"entities": [(27, 113, LABEL)]},
),
(
"In China Sales, receipts, or value of shipments of firms ",
{"entities": [(9, 56, LABEL)]},
),
(
"By census tract in the United States life expectancy ",
{"entities": [(37, 52, LABEL)]},
),
(
"In 1972 federal government expenditure (per capita) in the UK by township ",
{"entities": [(8, 51, LABEL)]},
),
(
"In 2011 in the United States by county Annual payroll ",
{"entities": [(39, 53, LABEL)]},
),
(
"In South Korea number of earthquake ",
{"entities": [(15, 35, LABEL)]},
),
(
"By census tract Production workers annual wages of Bakeries and tortilla manufacturing ",
{"entities": [(16, 86, LABEL)]},
),
(
"In 1951 Elementary-secondary revenue from federal sources ",
{"entities": [(8, 57, LABEL)]},
),
(
"By census tract Households with male householder, no wife present, family ",
{"entities": [(16, 73, LABEL)]},
),
(
"Households with female householder, no husband present, family in 1953 by county in the United States ",
{"entities": [(0, 62, LABEL)]},
),
(
"Elementary-secondary revenue from special education in the UK by county ",
{"entities": [(0, 51, LABEL)]},
),
(
"By township percent of houses with annual income of $300,000 and over in China ",
{"entities": [(12, 69, LABEL)]},
),
(
"By township Median household income ",
{"entities": [(12, 35, LABEL)]},
),
(
"In 2014 by county difference in population density of dentist ",
{"entities": [(18, 61, LABEL)]},
),
(
"Capital outlay of elementary-secondary expenditure by state in Canada in 2018 ",
{"entities": [(0, 50, LABEL)]},
),
(
"By state in China Household income in 1951 ",
{"entities": [(18, 34, LABEL)]},
),
(
"In 2008 by census tract Elementary-secondary revenue from local sources in China ",
{"entities": [(24, 71, LABEL)]},
),
(
"Percent of households above $200k by township in France in 1975 ",
{"entities": [(0, 33, LABEL)]},
),
(
"Average hours per day by men spent on Job search and interviewing by census tract ",
{"entities": [(0, 65, LABEL)]},
),
(
"In China Annual payroll of Other nonmetallic mineral product manufacturing by state ",
{"entities": [(9, 74, LABEL)]},
),
(
"By county Estimated annual sales for Pharmacies & drug stores in 1991 ",
{"entities": [(10, 61, LABEL)]},
),
(
"In Canada Elementary-secondary revenue from parent government contributions ",
{"entities": [(10, 75, LABEL)]},
),
(
"Total revenue of governments by state ",
{"entities": [(0, 28, LABEL)]},
),
(
"Number of firms in 1983 by state ",
{"entities": [(0, 15, LABEL)]},
),
(
"By census tract Sales, receipts, or value of shipments of firms in 2012 ",
{"entities": [(16, 63, LABEL)]},
),
(
"In the United States Age of householder ",
{"entities": [(21, 39, LABEL)]},
),
(
"In 1985 in France energy consumption (per capita) ",
{"entities": [(18, 49, LABEL)]},
),
(
"In the UK Annual payroll in 1967 by county ",
{"entities": [(10, 24, LABEL)]},
),
(
"Annual payroll by township in the United States in 2007 ",
{"entities": [(0, 14, LABEL)]},
),
(
"GDP (nominal or ppp) in France ",
{"entities": [(0, 20, LABEL)]},
),
(
"In France Family households (families) by county ",
{"entities": [(10, 38, LABEL)]},
),
(
"Nonfamily households in China by census tract ",
{"entities": [(0, 20, LABEL)]},
),
(
"In 2002 Elementary-secondary revenue from school lunch charges in Canada ",
{"entities": [(8, 62, LABEL)]},
),
(
"In 2004 by census tract Average percent of time engaged in by womenGovernment services ",
{"entities": [(24, 86, LABEL)]},
),
(
"Estimated annual sales for total (excl. gasoline stations) in 2001 by state ",
{"entities": [(0, 58, LABEL)]},
),
(
"Annual payroll of Primary metal manufacturing in 1983 in the United States ",
{"entities": [(0, 45, LABEL)]},
),
(
"In the UK Annual payroll of Converted paper product manufacturing ",
{"entities": [(10, 65, LABEL)]},
),
(
"In 2007 Total cost of materials of Clay product and refractory manufacturing ",
{"entities": [(8, 76, LABEL)]},
),
(
"By county burglary per 1000 household ",
{"entities": [(10, 37, LABEL)]},
),
(
"By state Number of employees of Nonferrous metal (except aluminum) production and processing in 1976 ",
{"entities": [(9, 92, LABEL)]},
),
(
"In the UK annual average temperature in 1950 by state ",
{"entities": [(10, 36, LABEL)]},
),
(
"Estimated annual sales for Sporting goods hobby, musical instrument, & book stores in the United States in 1989 ",
{"entities": [(0, 82, LABEL)]},
),
(
"Annual payroll by township in 1968 in Canada ",
{"entities": [(0, 14, LABEL)]},
),
(
"Number of libraries in 2016 ",
{"entities": [(0, 19, LABEL)]},
),
(
"By census tract in Canada number of libraries ",
{"entities": [(26, 45, LABEL)]},
),
(
"In 2001 Married-couple family ",
{"entities": [(8, 29, LABEL)]},
),
(
"In 1982 by census tract Elementary-secondary revenue from local sources in China ",
{"entities": [(24, 71, LABEL)]},
),
(
"In 2019 Sales, receipts, or value of shipments of firms in France ",
{"entities": [(8, 55, LABEL)]},
),
(
"In the United States annual average precipitation by county ",
{"entities": [(21, 49, LABEL)]},
),
(
"Family households (families) in 2020 ",
{"entities": [(0, 28, LABEL)]},
),
(
"Percent change of divorced in China in 2015 by census tract ",
{"entities": [(0, 26, LABEL)]},
),
(
"By state in Canada Married-couple family in 2014 ",
{"entities": [(19, 40, LABEL)]},
),
(
"Estimated annual sales for Pharmacies & drug stores in 1976 by state ",
{"entities": [(0, 51, LABEL)]},
),
(
"In South Korea by census tract in 2011 Estimated annual sales for Building material & garden eq. & supplies dealers ",
{"entities": [(39, 115, LABEL)]},
),
(
"By census tract Number of paid employees in France in 2016 ",
{"entities": [(16, 40, LABEL)]},
),
(
"In 1958 in the United States difference in number of people of death among children under 5 due to pediatric cancer ",
{"entities": [(29, 115, LABEL)]},
),
(
"Difference in number of people of divorced in South Korea by census tract ",
{"entities": [(0, 42, LABEL)]},
),
(
"In the UK Total cost of materials of Resin, synthetic rubber, and artificial synthetic fibers and filaments manufacturing ",
{"entities": [(10, 121, LABEL)]},
),
(
"Elementary-secondary revenue from special education by township in China ",
{"entities": [(0, 51, LABEL)]},
),
(
"Exports value of firms in South Korea by township ",
{"entities": [(0, 22, LABEL)]},
),
(
"By state in 2006 Insurance benefits and repayments of governments ",
{"entities": [(17, 65, LABEL)]},
),
(
"In Canada gross domestic income (nominal or ppp) per capita ",
{"entities": [(10, 59, LABEL)]},
),
(
"In 2020 life expectancy ",
{"entities": [(8, 23, LABEL)]},
),
(
"In South Korea difference in population density of people with a bachelor's degree or higher in 1984 ",
{"entities": [(15, 92, LABEL)]},
),
(
"In the UK by township Average hours per day by men spent on Kitchen and food cleanup ",
{"entities": [(22, 84, LABEL)]},
),
(
"By state in the United States Age of householder ",
{"entities": [(30, 48, LABEL)]},
),
(
"In 2009 Production workers average for year of Plastics product manufacturing by county ",
{"entities": [(8, 77, LABEL)]},
),
(
"In 1993 Average percent of time engaged in by womenCaring for nonhousehold adults by state ",
{"entities": [(8, 81, LABEL)]},
),
(
"In 1993 in South Korea General expenditure of governments by census tract ",
{"entities": [(23, 57, LABEL)]},
),
(
"In the UK by state Total value of shipments and receipts for services of Nonmetallic mineral product manufacturing ",
{"entities": [(19, 114, LABEL)]},
),
(
"By county difference in number of people of people enrolled in College or graduate school ",
{"entities": [(10, 89, LABEL)]},
),
(
"By census tract License plate vanitization rate in 1968 ",
{"entities": [(16, 47, LABEL)]},
),
(
"In 2007 in the United States number of pedestrian accidents ",
{"entities": [(29, 59, LABEL)]},
),
(
"In the UK GDP (nominal or ppp) per capita in 1960 by state ",
{"entities": [(10, 41, LABEL)]},
),
(
"By township percent of houses with annual income of $50,000 and less in 1972 ",
{"entities": [(12, 68, LABEL)]},
),
(
"By census tract in the UK Household income ",
{"entities": [(26, 42, LABEL)]},
),
(
"In 2009 by township Estimated annual sales for Gasoline stations in South Korea ",
{"entities": [(20, 64, LABEL)]},
),
(
"Population density of American Indian and Alaska Native in 2017 by census tract ",
{"entities": [(0, 55, LABEL)]},
),
(
"In the United States in 1992 Estimated annual sales for Building material & garden eq. & supplies dealers by state ",
{"entities": [(29, 105, LABEL)]},
),
(
"In the UK by state Average hours per day spent on Kitchen and food cleanup in 2005 ",
{"entities": [(19, 74, LABEL)]},
),
(
"Number of earthquake in 2003 ",
{"entities": [(0, 20, LABEL)]},
),
(
"In France number of earthquake ",
{"entities": [(10, 30, LABEL)]},
),
(
"In the UK in 1980 Total cost of materials of Alumina and aluminum production and processing ",
{"entities": [(18, 91, LABEL)]},
),
(
"In Canada in 1977 Estimated annual sales for Furniture stores ",
{"entities": [(18, 61, LABEL)]},
),
(
"In 2005 Utility expenditure of governments ",
{"entities": [(8, 42, LABEL)]},
),
(
"By township Population density of black or African American in the UK in 2017 ",
{"entities": [(12, 59, LABEL)]},
),
(
"By county Total households in 1998 ",
{"entities": [(10, 26, LABEL)]},
),
(
"In Canada in 1992 Annual payroll ",
{"entities": [(18, 32, LABEL)]},
),
(
"By county Average year built ",
{"entities": [(10, 28, LABEL)]},
),
(
"Number of paid employees in 2014 ",
{"entities": [(0, 24, LABEL)]},
),
(
"In France by census tract in 1968 Average monthly housing cost as percentage of income ",
{"entities": [(34, 86, LABEL)]},
),
(
"In 1987 in South Korea Exports value of firms ",
{"entities": [(23, 45, LABEL)]},
),
(
"By census tract Median household income in 1978 ",
{"entities": [(16, 39, LABEL)]},
),
(
"Household income in China ",
{"entities": [(0, 16, LABEL)]},
),
(
"In the UK unemployment rate ",
{"entities": [(10, 27, LABEL)]},
),
(
"In South Korea Average hours per day by men spent on Caring for nonhousehold adults in 1955 by census tract ",
{"entities": [(15, 83, LABEL)]},
),
(
"In 1966 number of fire points in the UK ",
{"entities": [(8, 29, LABEL)]},
),
(
"In the UK Age of householder in 2008 ",
{"entities": [(10, 28, LABEL)]},
),
(
"By county in the United States Estimated annual sales for Food & beverage stores ",
{"entities": [(31, 80, LABEL)]},
),
(
"In 1974 in the UK by state annual average precipitation ",
{"entities": [(27, 55, LABEL)]},
),
(
"Average poverty level for household by state ",
{"entities": [(0, 35, LABEL)]},
),
(
"By state Elementary-secondary revenue from property taxes in Canada in 1956 ",
{"entities": [(9, 57, LABEL)]},
),
(
"In 1958 divorce rate in China ",
{"entities": [(8, 20, LABEL)]},
),
(
"Number of employees of Other nonmetallic mineral product manufacturing in France in 1970 by county ",
{"entities": [(0, 70, LABEL)]},
),
(
"By census tract Elementary-secondary revenue from local sources in 1963 ",
{"entities": [(16, 63, LABEL)]},
),
(
"Average family size in 1992 ",
{"entities": [(0, 19, LABEL)]},
),
(
"Production workers average for year of Office furniture (including fixtures) manufacturing in the United States in 2008 by state ",
{"entities": [(0, 90, LABEL)]},
),
(
"By census tract Age of householder ",
{"entities": [(16, 34, LABEL)]},
),
(
"In China by county Average poverty level for household ",
{"entities": [(19, 54, LABEL)]},
),
(
"Average household size by township ",
{"entities": [(0, 22, LABEL)]},
),
(
"By state in 1961 number of McDonald's ",
{"entities": [(17, 37, LABEL)]},
),
(
"In Canada Intergovernmental expenditure of governments by census tract in 1973 ",
{"entities": [(10, 54, LABEL)]},
),
(
"Average percent of time engaged in by menGovernment services by state in 1959 ",
{"entities": [(0, 60, LABEL)]},
),
(
"By state in 1950 difference in number of people of People whose native language is Russian in the UK ",
{"entities": [(17, 90, LABEL)]},
),
(
"In France by state in 2003 Elementary-secondary revenue from general formula assistance ",
{"entities": [(27, 87, LABEL)]},
),
(
"Annual average precipitation in Canada ",
{"entities": [(0, 28, LABEL)]},
),
(
"Estimated annual sales for Shoe stores by census tract ",
{"entities": [(0, 38, LABEL)]},
),
(
"In the UK by county number of pedestrian accidents ",
{"entities": [(20, 50, LABEL)]},
),
(
"In China Family households with own children of the householder under 18 years ",
{"entities": [(9, 78, LABEL)]},
),
(
"In the United States Average hours per day by men spent on Other income-generating activities by census tract ",
{"entities": [(21, 93, LABEL)]},
),
(
"In the UK in 1983 Insurance trust revenue of governments by township ",
{"entities": [(18, 56, LABEL)]},
),
(
"In China by county Total cost of materials of Dairy product manufacturing ",
{"entities": [(19, 73, LABEL)]},
),
(
"Average percent of time engaged in by womenCaring for nonhousehold adults in China ",
{"entities": [(0, 73, LABEL)]},
),
(
"In Canada in 1998 by state Average hours per day by men spent on Caring for and helping nonhousehold adults ",
{"entities": [(27, 107, LABEL)]},
),
(
"In 1973 by county Average hours per day by women spent on Attending class in China ",
{"entities": [(18, 73, LABEL)]},
),
(
"Number of firms in South Korea ",
{"entities": [(0, 15, LABEL)]},
),
(
"In the UK in 2008 Annual payroll ",
{"entities": [(18, 32, LABEL)]},
),
(
"Estimated annual sales for total (excl. motor vehicle & parts & gasoline stations) by census tract in 1986 ",
{"entities": [(0, 82, LABEL)]},
),
(
"People living in poverty areas in South Korea ",
{"entities": [(0, 30, LABEL)]},
),
(
"Annual average temperature by census tract ",
{"entities": [(0, 26, LABEL)]},
),
(
"In France by township in 1995 Estimated annual sales for Auto & other motor veh. Dealers ",
{"entities": [(30, 88, LABEL)]},
),
(
"Average number of bedrooms of houses in the UK ",
{"entities": [(0, 36, LABEL)]},
),
(
"In 2008 by state Median household income ",
{"entities": [(17, 40, LABEL)]},
),
(
"Total value of shipments and receipts for services of Bakeries and tortilla manufacturing by census tract in 2003 ",
{"entities": [(0, 89, LABEL)]},
),
(
"In 1998 in France Percent of population of retailers of personal computer by township ",
{"entities": [(18, 73, LABEL)]},
),
(
"In 1975 in the United States Average hours per day by men spent on Attending religious services by census tract ",
{"entities": [(29, 95, LABEL)]},
),
(
"By census tract Average monthly housing cost as percentage of income in 1975 ",
{"entities": [(16, 68, LABEL)]},
),
(
"In Canada by township Elementary-secondary revenue from compensatory programs ",
{"entities": [(22, 77, LABEL)]},
),
(
"Annual payroll in the United States ",
{"entities": [(0, 14, LABEL)]},
),
(
"In 2000 Total cost of materials of Other furniture related product manufacturing in the UK by census tract ",
{"entities": [(8, 80, LABEL)]},
),
(
"By county in 1966 annual average temperature in China ",
{"entities": [(18, 44, LABEL)]},
),
(
"Interest on debt of governments in 2010 in France ",
{"entities": [(0, 31, LABEL)]},
),
(
"In 1959 Age of householder by state ",
{"entities": [(8, 26, LABEL)]},
),
(
"By census tract Insurance benefits and repayments of governments ",
{"entities": [(16, 64, LABEL)]},
),
(
"By state in the United States in 2004 annual average temperature ",
{"entities": [(38, 64, LABEL)]},
),
(
"In 2011 Average year built ",
{"entities": [(8, 26, LABEL)]},
),
(
"Estimated annual sales for New car dealers in 1988 ",
{"entities": [(0, 42, LABEL)]},
),
(
"In France by census tract Elementary-secondary revenue from state sources ",
{"entities": [(26, 73, LABEL)]},
),
(
"Households with one or more people under 18 years in China ",
{"entities": [(0, 49, LABEL)]},
),
(
"In 1950 by census tract number of fire points ",
{"entities": [(24, 45, LABEL)]},
),
(
"By state General expenditure of governments in France in 2006 ",
{"entities": [(9, 43, LABEL)]},
),
(
"Percent of population of people who are confirm to be infected by 2019-Nov Coronavirus in 1977 ",
{"entities": [(0, 86, LABEL)]},
),
(
"In the UK by census tract Number of firms ",
{"entities": [(26, 41, LABEL)]},
),
(
"Percent of households above $200k in 1979 by state in Canada ",
{"entities": [(0, 33, LABEL)]},
),
(
"By township in the United States Family households (families) ",
{"entities": [(33, 61, LABEL)]},
),
(
"Insurance trust revenue of governments by state in China ",
{"entities": [(0, 38, LABEL)]},
),
(
"In Canada Gross profit of companies ",
{"entities": [(10, 35, LABEL)]},
),
(
"In 1961 Household income in France ",
{"entities": [(8, 24, LABEL)]},
),
(
"In South Korea in 1965 Estimated annual sales for total by census tract ",
{"entities": [(23, 55, LABEL)]},
),
(
"In 1954 Cash and security holdings of governments ",
{"entities": [(8, 49, LABEL)]},
),
(
"Average monthly housing cost as percentage of income by township in France in 1975 ",
{"entities": [(0, 52, LABEL)]},
),
(
"By township in South Korea Age of householder ",
{"entities": [(27, 45, LABEL)]},
),
(
"Estimated annual sales for Clothing & clothing accessories stores in the United States in 2016 ",
{"entities": [(0, 65, LABEL)]},
),
(
"In 2007 Elementary-secondary revenue from compensatory programs in France ",
{"entities": [(8, 63, LABEL)]},
),
(
"By state in 1958 Married-couple family in South Korea ",
{"entities": [(17, 38, LABEL)]},
),
(
"Households with male householder, no wife present, family by census tract in 1997 ",
{"entities": [(0, 57, LABEL)]},
),
(
"By census tract Average percent of time engaged in by menInterior cleaning ",
{"entities": [(16, 74, LABEL)]},
),
(
"Capital outlay of elementary-secondary expenditure in 2014 in China ",
{"entities": [(0, 50, LABEL)]},
),
(
"By county Number of firms ",
{"entities": [(10, 25, LABEL)]},
),
(
"In 2019 in France Estimated annual sales for Warehouse clubs & supercenters ",
{"entities": [(18, 75, LABEL)]},
),
(
"By state in 1960 Number of paid employees in the United States ",
{"entities": [(17, 41, LABEL)]},
),
(
"By township gross domestic income (nominal or ppp) per capita ",
{"entities": [(12, 61, LABEL)]},
),
(
"Elementary-secondary expenditure in China by state in 2008 ",
{"entities": [(0, 32, LABEL)]},
),
(
"By census tract in France Average hours per day by men spent on Household and personal mail and messages ",
{"entities": [(26, 104, LABEL)]},
),
(
"Elementary-secondary revenue from general formula assistance in the United States ",
{"entities": [(0, 60, LABEL)]},
),
(
"Number of firms in Canada ",
{"entities": [(0, 15, LABEL)]},
),
(
"In 1961 by state in France poverty rate ",
{"entities": [(27, 39, LABEL)]},
),
(
"Income Taxes of governments in 2004 in South Korea by state ",
{"entities": [(0, 27, LABEL)]},
),
(
"Difference in number of people of people with a bachelor's degree or higher by census tract in 1977 ",
{"entities": [(0, 75, LABEL)]},
),
(
"Difference in population density of dentist in the UK ",
{"entities": [(0, 43, LABEL)]},
),
(
"Annual payroll of Manufacturing in Canada by township ",
{"entities": [(0, 31, LABEL)]},
),
(
"By state in the United States in 1987 Family households with own children of the householder under 18 years ",
{"entities": [(38, 107, LABEL)]},
),
(
"In China by township Annual payroll ",
{"entities": [(21, 35, LABEL)]},
),
(
"In the United States Corporate income tax of governments in 2002 by state ",
{"entities": [(21, 56, LABEL)]},
),
(
"Sales, receipts, or value of shipments of firms in the UK ",
{"entities": [(0, 47, LABEL)]},
),
(
"By state Race diversity index in 1982 in the United States ",
{"entities": [(9, 29, LABEL)]},
),
(
"In 2016 by state in South Korea annual average precipitation ",
{"entities": [(32, 60, LABEL)]},
),
(
"By county Elementary-secondary revenue from local government in 1950 ",
{"entities": [(10, 60, LABEL)]},
),
(
"In 2001 by state number of people of White ",
{"entities": [(17, 42, LABEL)]},
),
(
"By township number of people of people enrolled in Nursery school, people enrolled in preschool ",
{"entities": [(12, 95, LABEL)]},
),
(
"In France in 1955 Estimated annual sales for Building material & garden eq. & supplies dealers ",
{"entities": [(18, 94, LABEL)]},
),
(
"In South Korea in 1973 Family households with own children of the householder under 18 years by county ",
{"entities": [(23, 92, LABEL)]},
),
(
"By township in 1976 in the United States Exports value of firms ",
{"entities": [(41, 63, LABEL)]},
),
(
"In France Elementary-secondary revenue from compensatory programs ",
{"entities": [(10, 65, LABEL)]},
),
(
"In China Total revenue of governments ",
{"entities": [(9, 37, LABEL)]},
),
(
"In 1958 in Canada by township Percent change of people enrolled in Nursery school, people enrolled in preschool ",
{"entities": [(30, 111, LABEL)]},
),
(
"In 1952 in the UK Elementary-secondary revenue from property taxes ",
{"entities": [(18, 66, LABEL)]},
),
(
"In 2019 by township Current spending of elementary-secondary expenditure in the United States ",
{"entities": [(20, 72, LABEL)]},
),
(
"GDP (nominal or ppp) per capita in 1954 by state ",
{"entities": [(0, 31, LABEL)]},
),
(
"By state Average square footage of houses in China in 1989 ",
{"entities": [(9, 41, LABEL)]},
),
(
"By county in 2020 Household income in France ",
{"entities": [(18, 34, LABEL)]},
),
(
"In 1951 Average hours per day by women spent on Helping household adults by township ",
{"entities": [(8, 72, LABEL)]},
),
(
"In 1955 Average percent of time engaged in by menCaring for nonhousehold adults ",
{"entities": [(8, 79, LABEL)]},
),
(
"Annual average precipitation in South Korea ",
{"entities": [(0, 28, LABEL)]},
),
(
"In the United States in 1963 gross domestic income (nominal or ppp) per capita ",
{"entities": [(29, 78, LABEL)]},
),
(
"Elementary-secondary revenue from school lunch charges by census tract in 1989 in the UK ",
{"entities": [(0, 54, LABEL)]},
),
(
"In the United States average price for honey per pound by county ",
{"entities": [(21, 54, LABEL)]},
),
(
"Average percent of time engaged in by womenReading for personal interest by census tract in 1955 ",
{"entities": [(0, 72, LABEL)]},
),
(
"In the UK Households with female householder, no husband present, family by township ",
{"entities": [(10, 72, LABEL)]},
),
(
"In Canada annual average precipitation by state ",
{"entities": [(10, 38, LABEL)]},
),
(
"In the UK Households with one or more people 65 years and over by township in 1958 ",
{"entities": [(10, 62, LABEL)]},
),
(
"In 1986 Annual payroll of Plastics product manufacturing by county ",
{"entities": [(8, 56, LABEL)]},
),
(
"Interest on general debt of governments in France ",
{"entities": [(0, 39, LABEL)]},
),
(
"By county annual average precipitation ",
{"entities": [(10, 38, LABEL)]},
),
(
"Population density of women that were screened for breast and cervical cancer by jurisdiction in 1971 in Canada ",
{"entities": [(0, 93, LABEL)]},
),
(
"In China Elementary-secondary expenditure in 2014 by census tract ",
{"entities": [(9, 41, LABEL)]},
),
(
"Number of fire points in the UK by county in 2003 ",
{"entities": [(0, 21, LABEL)]},
),
(
"By census tract Total households ",
{"entities": [(16, 32, LABEL)]},
),
(
"In 1969 in South Korea by census tract Sales, receipts, or value of shipments of firms ",
{"entities": [(39, 86, LABEL)]},
),
(
"In China crime rate ",
{"entities": [(9, 19, LABEL)]},
),
(
"In the United States Number of paid employees in 2009 ",
{"entities": [(21, 45, LABEL)]},
),
(
"Fertility rate in Canada in 1972 by county ",
{"entities": [(0, 14, LABEL)]},
),
(
"In the United States Estimated annual sales for Shoe stores by census tract in 1991 ",
{"entities": [(21, 59, LABEL)]},
),
(
"By township Income Taxes of governments in the UK in 2012 ",
{"entities": [(12, 39, LABEL)]},
),
(
"Production workers annual wages of Printing and related support activities by township ",
{"entities": [(0, 74, LABEL)]},
),
(
"In 1962 by county Average hours per day by men spent on Activities related to household children health ",
{"entities": [(18, 103, LABEL)]},
),
(
"In the United States by state Liquor stores revenue of governments in 1959 ",
{"entities": [(30, 66, LABEL)]},
),
(
"By township in France economic growth rate in 2018 ",
{"entities": [(22, 42, LABEL)]},
),
(
"By census tract Estimated annual sales for Home furnishings stores ",
{"entities": [(16, 66, LABEL)]},
),
(
"Current charge of governments in China in 1976 by township ",
{"entities": [(0, 29, LABEL)]},
),
(
"In 1982 percent of houses with annual income of $50,000 and less ",
{"entities": [(8, 64, LABEL)]},
),
(
"Average percent of time engaged in Travel related to caring for and helping nonhousehold membership in China ",
{"entities": [(0, 99, LABEL)]},
),
(
"In South Korea Capital outlay of elementary-secondary expenditure in 1981 ",
{"entities": [(15, 65, LABEL)]},
),
(
"In 1957 Total expenditure of governments ",
{"entities": [(8, 40, LABEL)]},
),
(
"In the UK GDP (nominal or ppp) per capita in 1993 ",
{"entities": [(10, 41, LABEL)]},
),
(
"By county import and export statistics ",
{"entities": [(10, 38, LABEL)]},
),
(
"By census tract Age of householder in 1959 ",
{"entities": [(16, 34, LABEL)]},
),
(
"Salaries and wages of governments by county ",
{"entities": [(0, 33, LABEL)]},
),
(
"In 2012 Average monthly housing cost as percentage of income by state in the United States ",
{"entities": [(8, 60, LABEL)]},
),
(
"In 1959 in the United States by census tract Production workers annual wages of Footwear manufacturing ",
{"entities": [(45, 102, LABEL)]},
),
(
"By county in 1984 sale amounts of beer in France ",
{"entities": [(18, 38, LABEL)]},
),
(
"By census tract Total households ",
{"entities": [(16, 32, LABEL)]},
),
(
"In 1966 in France Households with householder living alone ",
{"entities": [(18, 58, LABEL)]},
),
(
"In 2011 Average poverty level for household in South Korea ",
{"entities": [(8, 43, LABEL)]},
),
(
"In Canada percent of houses with annual income of $50,000 and less ",
{"entities": [(10, 66, LABEL)]},
),
(
"By census tract in 1989 Average percent of time engaged in by womenActivities related to household children health ",
{"entities": [(24, 114, LABEL)]},
),
(
"In the UK by census tract in 1983 License Taxes of governments ",
{"entities": [(34, 62, LABEL)]},
),
(
"Exports value of firms in 1999 ",
{"entities": [(0, 22, LABEL)]},
),
(
"In China in 2008 Average number of bedrooms of houses ",
{"entities": [(17, 53, LABEL)]},
),
(
"In the UK by county in 2015 people living in poverty areas ",
{"entities": [(28, 58, LABEL)]},
),
(
"In France Average poverty level for household in 1973 ",
{"entities": [(10, 45, LABEL)]},
),
(
"In Canada Estimated annual sales for General merchandise stores by census tract in 1987 ",
{"entities": [(10, 63, LABEL)]},
),
(
"In 1964 Average monthly housing cost as percentage of income ",
{"entities": [(8, 60, LABEL)]},
),
(
"Average poverty level for household by county in France ",
{"entities": [(0, 35, LABEL)]},
),
(
"Production workers average for year of Other transportation equipment manufacturing,Furniture and related product manufacturing in 1955 ",
{"entities": [(0, 127, LABEL)]},
),
(
"Total cost of materials of Nonmetallic mineral product manufacturing in France in 1996 ",
{"entities": [(0, 68, LABEL)]},
),
(
"By census tract in 1991 Age of householder ",
{"entities": [(24, 42, LABEL)]},
),
(
"General sales of governments in 2000 ",
{"entities": [(0, 28, LABEL)]},
),
(
"Difference in number of people of divorced in France by township in 1995 ",
{"entities": [(0, 42, LABEL)]},
),
(
"Average household size by state in South Korea in 2012 ",
{"entities": [(0, 22, LABEL)]},
),
(
"By township in the UK Average hours per day by women spent on Interior maintenance, repair, and decoration ",
{"entities": [(22, 106, LABEL)]},
),
(
"Total value of shipments and receipts for services of Beverage manufacturing in 1981 in Canada ",
{"entities": [(0, 76, LABEL)]},
),
(
"By state in 2001 Number of paid employees ",
{"entities": [(17, 41, LABEL)]},
),
(
"By township Average hours per day spent on Physical care for household children in the United States ",
{"entities": [(12, 79, LABEL)]},
),
(
"By census tract Average hours per day by men spent on Kitchen and food cleanup in China ",
{"entities": [(16, 78, LABEL)]},
),
(
"In 1991 difference in number of people of people who are confirm to be infected by 2019-Nov Coronavirus ",
{"entities": [(8, 103, LABEL)]},
),
(
"Number of people of Jewish by state ",
{"entities": [(0, 26, LABEL)]},
),
(
"Family households with own children of the householder under 18 years in 1981 in the UK ",
{"entities": [(0, 69, LABEL)]},
),
(
"GDP (nominal or ppp) per capita in China ",
{"entities": [(0, 31, LABEL)]},
),
(
"By census tract Estimated annual sales for Gasoline stations ",
{"entities": [(16, 60, LABEL)]},
),
(
"By county Married-couple family ",
{"entities": [(10, 31, LABEL)]},
),
(
"In 2006 in Canada by county Estimated annual sales for Warehouse clubs & supercenters ",
{"entities": [(28, 85, LABEL)]},
),
(
"In the United States in 1993 Exports value of firms ",
{"entities": [(29, 51, LABEL)]},
),
(
"Burglary per 1000 household in 1972 by state ",
{"entities": [(0, 27, LABEL)]},
),
(
"Average square footage of houses in France ",
{"entities": [(0, 32, LABEL)]},
),
(
"In China in 2002 Percent of population of people who are confirm to be infected by 2019-Nov Coronavirus ",
{"entities": [(17, 103, LABEL)]},
),
(
"In 1978 infant mortality rate ",
{"entities": [(8, 29, LABEL)]},
),
(
"In France freedom index ",
{"entities": [(10, 23, LABEL)]},
),
(
"In the UK Direct expenditure of governments by county ",
{"entities": [(10, 43, LABEL)]},
),
(
"In the United States by township Number of paid employees in 1956 ",
{"entities": [(33, 57, LABEL)]},
),
(
"Estimated annual sales for Auto parts in the UK ",
{"entities": [(0, 37, LABEL)]},
),
(
"By state Average monthly housing cost as percentage of income in France ",
{"entities": [(9, 61, LABEL)]},
),
(
"Diabetes rate by census tract in 1974 ",
{"entities": [(0, 13, LABEL)]},
),
(
"Annual average precipitation in 1974 by township ",
{"entities": [(0, 28, LABEL)]},
),
(
"By township in South Korea life expectancy ",
{"entities": [(27, 42, LABEL)]},
),
(
"In South Korea General sales of governments ",
{"entities": [(15, 43, LABEL)]},
),
(
"Production workers average for year of Nonferrous metal (except aluminum) production and processing in 2014 by census tract ",
{"entities": [(0, 99, LABEL)]},
),
(
"Average number of bedrooms of houses by census tract in Canada ",
{"entities": [(0, 36, LABEL)]},
),
(
"Import and export statistics by county in China in 2014 ",
{"entities": [(0, 28, LABEL)]},
),
(
"In the United States in 2002 by state federal government expenditure (per capita) ",
{"entities": [(38, 81, LABEL)]},
),
(
"Poverty rate in China by county ",
{"entities": [(0, 12, LABEL)]},
),
(
"In 2017 by township Estimated annual sales for Beer, wine & liquor stores in China ",
{"entities": [(20, 73, LABEL)]},
),
(
"By township in 1992 Average percent of time engaged in by womenAttending household children events ",
{"entities": [(20, 98, LABEL)]},
),
(
"Number of firms in 2007 ",
{"entities": [(0, 15, LABEL)]},
),
(
"Sale amounts of beer by township ",
{"entities": [(0, 20, LABEL)]},
),
(
"In the UK Production workers average for year of Communications equipment manufacturing in 2017 by census tract ",
{"entities": [(10, 87, LABEL)]},
),
(
"In France Estimated annual sales for Family clothing stores by census tract in 1986 ",
{"entities": [(10, 59, LABEL)]},
),
(
"By state Assistance and subsidies of governments in 1994 ",
{"entities": [(9, 48, LABEL)]},
),
(
"By county Average percent of time engaged in by womenTravel related to caring for and helping household members in South Korea ",
{"entities": [(10, 111, LABEL)]},
),
(
"In 1997 by census tract in the United States Households with householder living alone ",
{"entities": [(45, 85, LABEL)]},
),
(
"By state General revenue of governments in 2005 in the United States ",
{"entities": [(9, 39, LABEL)]},
),
(
"Elementary-secondary revenue from vocational programs by state ",
{"entities": [(0, 53, LABEL)]},
),
(
"Renter occupied by state ",
{"entities": [(0, 15, LABEL)]},
),
(
"Annual average precipitation by census tract ",
{"entities": [(0, 28, LABEL)]},
),
(
"In 1999 Average number of bedrooms of houses by township in the United States ",
{"entities": [(8, 44, LABEL)]},
),
(
"In 1992 Total capital expenditures of Cut and sew apparel manufacturing ",
{"entities": [(8, 71, LABEL)]},
),
(
"By state Elementary-secondary revenue from special education in the United States ",
{"entities": [(9, 60, LABEL)]},
),
(
"Average percent of time engaged in by menSports, exercise, and recreation by county ",
{"entities": [(0, 73, LABEL)]},
),
(
"Median household income by state in 1951 in the United States ",
{"entities": [(0, 23, LABEL)]},
),
(
"By state in 1958 in the United States Total cost of materials of Beverage and tobacco product manufacturing ",
{"entities": [(38, 107, LABEL)]},
),
(
"In Canada Average percent of time engaged in Travel related to household activities in 1963 by township ",
{"entities": [(10, 83, LABEL)]},
),
(
"By county in France in 2009 Average number of bedrooms of houses ",
{"entities": [(28, 64, LABEL)]},
),
(
"Average number of bedrooms of houses by township in 1951 in Canada ",
{"entities": [(0, 36, LABEL)]},
),
(
"General expenditure of governments in the United States in 1988 ",
{"entities": [(0, 34, LABEL)]},
),
(
"Number of fire points in China in 1960 by township ",
{"entities": [(0, 21, LABEL)]},
),
(
"In South Korea Average square footage of houses in 2010 by township ",
{"entities": [(15, 47, LABEL)]},
),
(
"Average hours per day by men spent on Financial management by census tract ",
{"entities": [(0, 58, LABEL)]},
),
(
"In 2012 in China average price for honey per pound by township ",
{"entities": [(17, 50, LABEL)]},
),
(
"Percent of population of people who are confirm to be infected by 2019-Nov Coronavirus in 2007 by county in the UK ",
{"entities": [(0, 86, LABEL)]},
),
(
"In 1966 Average number of bedrooms of houses ",
{"entities": [(8, 44, LABEL)]},
),
(
"In Canada by county in 1994 suicide rate ",
{"entities": [(28, 40, LABEL)]},
),
(
"In China Elementary-secondary revenue from local government ",
{"entities": [(9, 59, LABEL)]},
),
(
"Elementary-secondary revenue from parent government contributions in South Korea in 1953 ",
{"entities": [(0, 65, LABEL)]},
),
(
"Number of patent per capita in South Korea by township ",
{"entities": [(0, 27, LABEL)]},
),
(
"In Canada Average family size in 1989 by census tract ",
{"entities": [(10, 29, LABEL)]},
),
(
"In 2003 in France Elementary-secondary revenue ",
{"entities": [(18, 46, LABEL)]},
),
(
"By township Average hours per day by women spent on Travel related to caring for and helping household members in Canada ",
{"entities": [(12, 110, LABEL)]},
),
(
"Average square footage of houses by township ",
{"entities": [(0, 32, LABEL)]},
),
(
"Elementary-secondary revenue from vocational programs in 1977 by state in Canada ",
{"entities": [(0, 53, LABEL)]},
),
(
"Difference in race diversity by state ",
{"entities": [(0, 28, LABEL)]},
),
(
"In Canada number of McDonald's by state in 1955 ",
{"entities": [(10, 30, LABEL)]},
),
(
"Households with one or more people under 18 years by state ",
{"entities": [(0, 49, LABEL)]},
),
(
"Individual income tax of governments in Canada ",
{"entities": [(0, 36, LABEL)]},
),
(
"By county Estimated annual sales for Home furnishings stores ",
{"entities": [(10, 60, LABEL)]},
),
(
"By state Other taxes of governments ",
{"entities": [(9, 35, LABEL)]},
),
(
"Estimated annual sales for Pharmacies & drug stores by census tract in the United States ",
{"entities": [(0, 51, LABEL)]},
),
(
"Price of land by township ",
{"entities": [(0, 13, LABEL)]},
),
(
"Average year built in 2017 in South Korea ",
{"entities": [(0, 18, LABEL)]},
),
(
"Elementary-secondary revenue from transportation programs in 1987 ",
{"entities": [(0, 57, LABEL)]},
),
(
"Estimated annual sales for Electronics & appliance stores in France by state ",
{"entities": [(0, 57, LABEL)]},
),
(
"In 1983 Households with householder living alone ",
{"entities": [(8, 48, LABEL)]},
),
(
"By census tract in 2020 Production workers annual wages of Paper manufacturing in the United States ",
{"entities": [(24, 78, LABEL)]},
),
(
"Number of people of people with elementary occupation by county in Canada ",
{"entities": [(0, 53, LABEL)]},
),
(
"By county in China in 1979 Intergovernmental expenditure of governments ",
{"entities": [(27, 71, LABEL)]},
),
(
"In Canada Capital outlay of governments ",
{"entities": [(10, 39, LABEL)]},
),
(
"Number of fire points by township in China ",
{"entities": [(0, 21, LABEL)]},
),
(
"By county in 1976 Family households (families) ",
{"entities": [(18, 46, LABEL)]},
),
(
"By state Average hours per day by men spent on Medical and care services in France ",
{"entities": [(9, 72, LABEL)]},
),
(
"Difference in number of people of women that were screened for breast and cervical cancer by jurisdiction in 1958 ",
{"entities": [(0, 105, LABEL)]},
),
(
"Insurance trust expenditure of governments in France ",
{"entities": [(0, 42, LABEL)]},
),
(
"Age of householder in 1985 in China ",
{"entities": [(0, 18, LABEL)]},
),
(
"In 1969 Annual payroll of Grain and oilseed milling ",
{"entities": [(8, 51, LABEL)]},
),
(
"Crime rate in China ",
{"entities": [(0, 10, LABEL)]},
),
(
"In 2018 Annual payroll of Office furniture (including fixtures) manufacturing by census tract ",
{"entities": [(8, 77, LABEL)]},
),
(
"Households with female householder, no husband present, family in 1990 ",
{"entities": [(0, 62, LABEL)]},
),
(
"Freedom index by township ",
{"entities": [(0, 13, LABEL)]},
),
(
"In France in 2010 Annual payroll by census tract ",
{"entities": [(18, 32, LABEL)]},
),
(
"In the United States in 1981 by township Age of householder ",
{"entities": [(41, 59, LABEL)]},
),
(
"In 1968 in France Salaries and wages of governments by township ",
{"entities": [(18, 51, LABEL)]},
),
(
"Total value of shipments and receipts for services of Rubber product manufacturing by state in 1967 ",
{"entities": [(0, 82, LABEL)]},
),
(
"By township in 1956 Insurance trust revenue of governments ",
{"entities": [(20, 58, LABEL)]},
),
(
"By county Estimated annual sales for Electronics & appliance stores in France ",
{"entities": [(10, 67, LABEL)]},
),
(
"Households with householder living alone in 1950 ",
{"entities": [(0, 40, LABEL)]},
),
(
"Social vulnerability index in 1982 in the United States ",
{"entities": [(0, 26, LABEL)]},
),
(
"In the UK License plate vanitization rate by township ",
{"entities": [(10, 41, LABEL)]},
),
(
"In 1960 by state Elementary-secondary revenue from vocational programs ",
{"entities": [(17, 70, LABEL)]},
),
(
"Average monthly housing cost as percentage of income by county ",
{"entities": [(0, 52, LABEL)]},
),
(
"In 1955 Production workers annual wages of Footwear manufacturing by census tract ",
{"entities": [(8, 65, LABEL)]},
),
(
"In France by county in 1990 Average percent of time engaged in by menAppliances, tools, and toys ",
{"entities": [(28, 96, LABEL)]},
),
(
"In South Korea in 2015 by census tract Corporate income tax of governments ",
{"entities": [(39, 74, LABEL)]},
),
(
"Population density of dentist by census tract ",
{"entities": [(0, 29, LABEL)]},
),
(
"In 1950 number of fire points by state ",
{"entities": [(8, 29, LABEL)]},
),
(
"In the UK Average hours per day spent on Purchasing goods and services by county in 1973 ",
{"entities": [(10, 70, LABEL)]},
),
(
"By township in 1950 in South Korea Annual payroll ",
{"entities": [(35, 49, LABEL)]},
),
(
"Adult obesity by state in 1973 ",
{"entities": [(0, 13, LABEL)]},
),
(
"By state Total capital expenditures of Manufacturing and reproducing magnetic and optical media in Canada in 1983 ",
{"entities": [(9, 95, LABEL)]},
),
(
"By county in the UK Average hours per day by women spent on Consumer goods purchases ",
{"entities": [(20, 84, LABEL)]},
),
(
"Estimated annual sales for Health & personal care stores in 2009 by state in South Korea ",
{"entities": [(0, 56, LABEL)]},
),
(
"In the UK in 1955 number of cell phones per 100 person ",
{"entities": [(18, 54, LABEL)]},
),
(
"Number of firms in the United States ",
{"entities": [(0, 15, LABEL)]},
),
(
"By township rate of male in Canada ",
{"entities": [(12, 24, LABEL)]},
),
(
"In 1961 Estimated annual sales for Food services & drinking places by state in the United States ",
{"entities": [(8, 66, LABEL)]},
),
(
"By state Average monthly housing cost as percentage of income ",
{"entities": [(9, 61, LABEL)]},
),
(
"By township in 1969 Estimated annual sales for Electronics & appliance stores ",
{"entities": [(20, 77, LABEL)]},
),
(
"By census tract in France in 1972 Household income ",
{"entities": [(34, 50, LABEL)]},
),
(
"In 1964 Household income ",
{"entities": [(8, 24, LABEL)]},
),
(
"In the UK Estimated annual sales for Other general merch. Stores by township ",
{"entities": [(10, 64, LABEL)]},
),
(
"By state Average percent of time engaged in Vehicles in South Korea in 2004 ",
{"entities": [(9, 52, LABEL)]},
)]

@plac.annotations(
    model=(nlpTest, "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model="en_core_web_sm", new_model_name="animal", output_dir=None, n_iter=100):
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

    ner.add_label(LABEL)  # add new entity label to entity recognizer
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
    for test_text in test_text_list:
        doc = nlp(test_text)
        print("Entities in '%s'" % test_text)
        for ent in doc.ents:
            print(ent.label_, ent.text)

    # save model to output directory
    with open('C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Name Entity Recognition\\en_core_web_sm_THEME.pkl', 'wb') as f:
        pickle.dump(nlp,f)

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
