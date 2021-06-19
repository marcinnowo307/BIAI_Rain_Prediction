# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 18:00:59 2021

@author: Marcinek
"""
import tensorflow as tf
import sys
import pandas as pd
import numpy as np
import sklearn as sk

from neural_network.network import create_model
from sklearn import preprocessing
from pickle import load

def load_json_data(json_name):
    return pd.read_json(json_name)

def preprocess_data(df):
    #drop columns
    df['Date'] = pd.to_datetime(df["Date"])
    df['Month'] = df.Date.dt.month
    df = df.drop(['Date'], axis=1)
    
    # turn object cols to ints
    tmp = (df.dtypes == "object")
    object_columns = list(tmp[tmp].index)
    for i in object_columns:
        encoder = sk.preprocessing.LabelEncoder()
        encoder.classes_ = np.load("neural_network/encodings/{0}.npy".format(i), allow_pickle=True)
        df[i] = encoder.transform(df[i])
    
    #scale the values
    column_names = list(df.columns)
    standard_scaler = load(open('neural_network/scaling/scaler.pkl', 'rb'))
    df = standard_scaler.transform(df)
    df = pd.DataFrame(df, columns=column_names)
    return df

def predict(df):
    model = create_model()
    model.load_weights('neural_network/weights')
    pred = model.predict(df)
    return (pred > 0.5)

if __name__ == "__main__":  
    print("will it rain tomorrow?")
    
    # data = pd.read_csv("original_dataset/weatherAUS.csv")
    # data.drop( labels = 'RainTomorrow', axis = 1, inplace = True)
    # data.dropna(inplace = True)
    # data.reset_index(drop = True, inplace = True)
    
    # frame = data.loc[data['Date'] == "2009-01-22"]
    # frame = frame.loc[frame['Location'] == "Cobar"]
    # frame.to_json("input.json")
    data = pd.read_json(sys.argv[1])
    dateTimes = data['Date']
    
    data = preprocess_data(data)
    predictions = predict(data)
    
    output = pd.DataFrame()
    output['Date'] = dateTimes
    output['RainTomorrow'] = predictions
    
    if output['RainTomorrow'][0] == True:
        print('it will')
    else:
        print("it won't")