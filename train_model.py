# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:06:47 2021

@author: Marcinek
"""
import sys
sys.path.insert(1, "neural_network")

import tensorflow as tf #for ANN purposes
import pandas as pd #for loading csv files
from sklearn.model_selection import train_test_split
from network import create_model
from sklearn.metrics import classification_report
import numpy

numpy.random.seed(2137)

#loading and splitting data
data = pd.read_csv("clean dataset/weatherAUS.csv")
x = data.drop(['RainTomorrow'], axis=1)
y = data['RainTomorrow']
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 40)
data.info()


model = create_model()
model.fit(x_train, y_train, batch_size = 32, epochs = 20, validation_split = 0.2)
model.save_weights('neural_network/weights')


predicted = model.predict(x_test)
predicted = (predicted > 0.5)
print(classification_report(y_test, predicted))

