# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 18:00:59 2021

@author: Marcinek
"""
import pandas as pd
import numpy as np
import sklearn as sk

from neural_network.network import create_model
from sklearn import preprocessing
from pickle import load

def load_csv_data(csv_name):
    data = pd.read_csv(csv_name)
    return data

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
    data = load_csv_data("original_dataset/weatherAUS.csv")
    data.drop( labels = 'RainTomorrow', axis = 1, inplace = True)
    data.dropna(inplace = True)
    data.reset_index(drop = True, inplace = True)
    
    data = preprocess_data(data)
    predictions = predict(data)
    print('asfd')