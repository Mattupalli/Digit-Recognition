# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 18:16:01 2018

@author: nishant
"""

import numpy as np
import tensorflow as tf
import tflearn
import matplotlib.pyplot as plot
import tflearn.datasets.mnist as mt

trainX, trainY, testX, testY = mt.load_data(one_hot=True)

def build_model():
    tf.reset_default_graph()
    
    net=tflearn.input_data([None, 784])
    
    net = tflearn.fully_connected(net, 10, activation='ReLU')
    net = tflearn.fully_connected(net, 20, activation='ReLU')
    
    net = tflearn.fully_connected(net, 10, activation='softmax')
    net = tflearn.regression(net, optimizer='sgd', learning_rate=0.001, loss='categorical_crossentropy')
    
    model = tflearn.DNN(net)
    return model

model = build_model()
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=100, n_epoch=5)

predictions = np.array(model.predict(testX)).argmax(axis=1)

actual = testY.argmax(axis=1)
accuracy = np.mean(predictions == actual, axis=0)

print("The accuracy is:", accuracy)

    