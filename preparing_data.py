# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 23:15:58 2021

@author: Marcinek
"""

import matplotlib.pyplot as plt # for plotting
import pandas as pd #for loading csv files
import numpy
import sklearn as sk
from sklearn import preprocessing
from pathlib import Path #to manipulate files
from pickle import dump

#create a directory for the clean datasets
Path("clean_dataset").mkdir(exist_ok=True)      
Path("neural_network").mkdir(exist_ok=True)
Path("neural_network/scaling").mkdir(exist_ok=True)      
Path("neural_network/encodings").mkdir(exist_ok=True)            


#Load the original csv file
original = pd.read_csv("original_dataset/weatherAUS.csv")
original.head()
original.info()
original.isnull().sum()

# Datetime conversion
original['Date'] = pd.to_datetime(original["Date"])
original['Month'] = original.Date.dt.month

# if target variable is null, drop it
original.dropna(subset = ['RainTomorrow'], inplace = True)
original.reset_index(drop = True, inplace = True)

# fill categorical NaN values with mode
tmp = (original.dtypes == "object")
object_columns = list(tmp[tmp].index)
for i in object_columns:
    original[i].fillna(original[i].mode()[0] , inplace=True)


# fill continous NaN values with median
tmp = (original.dtypes == "float64")
float_columns = list(tmp[tmp].index)
for i in float_columns:
    original[i].fillna(original[i].median() , inplace=True)


#preprocessing data

# turn object cols to ints
for i in object_columns:
    encoder = sk.preprocessing.LabelEncoder()
    encoder.fit(original[i])
    numpy.save( 'neural_network/encodings/{0}'.format(i), encoder.classes_)
    original[i] = encoder.transform(original[i])
    #original[i] = encoder.fit_transform(original[i])
    

    
target = original['RainTomorrow']
features = original.drop(['RainTomorrow', 'Date'], axis=1)
print('Shape before deleting outliers ', features.shape)

#scale the values
column_names = list(features.columns)
standard_scaler = preprocessing.StandardScaler()
features = standard_scaler.fit_transform(features)
dump(standard_scaler, open('neural network/scaling/scaler.pkl', 'wb'))
#features = standard_scaler.transform(features)
features = pd.DataFrame(features, columns=column_names)


# find outliers and delete them
features['RainTomorrow'] = target

#delete the outliers using IQR
#for i in ['Cloud3pm', 'Rainfall', 'Evaporation', 'WindSpeed9am']:
#    Q1 = features[i].quantile(0.25)
#    Q3 = features[i].quantile(0.75)
#    IQR = Q3 - Q1
#    lower_limit, upper_limit = Q1 - 5*IQR, Q3 + 5*IQR
#    features = features[(features[i] >= lower_limit) & (features[i] <= upper_limit)]


# hand trimming the dataset
features = features[(features["MinTemp"]<2.3)&(features["MinTemp"]>-2.3)]
features = features[(features["MaxTemp"]<2.3)&(features["MaxTemp"]>-2)]
features = features[(features["Rainfall"]<4.5)]
features = features[(features["Evaporation"]<2.8)]
features = features[(features["Sunshine"]<2.1)]
features = features[(features["WindGustSpeed"]<4)&(features["WindGustSpeed"]>-4)]
features = features[(features["WindSpeed9am"]<4)]
features = features[(features["WindSpeed3pm"]<2.5)]
features = features[(features["Humidity9am"]>-3)]
features = features[(features["Humidity3pm"]>-2.2)]
features = features[(features["Pressure9am"]< 2)&(features["Pressure9am"]>-2.7)]
features = features[(features["Pressure3pm"]< 2)&(features["Pressure3pm"]>-2.7)]
features = features[(features["Cloud9am"]<1.8)]
features = features[(features["Cloud3pm"]<2)]
features = features[(features["Temp9am"]<2.3)&(features["Temp9am"]>-2)]
features = features[(features["Temp3pm"]<2.3)&(features["Temp3pm"]>-2)]

##################################################################
#plt.figure(figsize=(20,10))
#sns.boxplot(data = features)
#plt.xticks(rotation = 90)
#plt.show()
##################################################################

print('Shape after deleting outliers ', features.shape)
features.info()

features.to_csv("clean_dataset/weatherAUS.csv", index=False)
