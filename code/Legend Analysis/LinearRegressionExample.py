import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
# import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('C:\\Users\\jiali\\Desktop\\MapElementDetection\\code\\Legend Analysis\\winequality-red.csv')

dataset.isnull().any()

dataset = dataset.fillna(method='ffill')

X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']].values
y = dataset['quality'].values

print ('\n')