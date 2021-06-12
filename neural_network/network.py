# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:07:34 2021

@author: Marcinek
"""

import tensorflow as tf

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu', input_dim = 22),
        tf.keras.layers.Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'),
        tf.keras.layers.Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'),
#        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'),
#        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')
        ])
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.00009)
    model.compile(optimizer='adam',
        loss = 'binary_crossentropy',
        metrics = ['accuracy'])
    return model