# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:06:47 2021

@author: Marcinek
"""

import pandas as pd #for loading csv files
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from neural_network.network import create_model

np.random.seed(2137)

#loading and splitting data
data = pd.read_csv("clean_dataset/weatherAUS.csv")
x = data.drop(['RainTomorrow'], axis=1)
y = data['RainTomorrow']
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 40)

# train a model
model = create_model()
model.fit(x_train, y_train, batch_size = 32, epochs = 20, validation_split = 0.2)
model.save_weights('neural_network/weights')

# check accuracy
predicted = model.predict(x_test)
predicted = (predicted > 0.5)
print(classification_report(y_test, predicted))

