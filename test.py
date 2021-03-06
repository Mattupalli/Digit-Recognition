# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 18:16:01 2018
@author: nishant
"""

import numpy as np
import tensorflow as tf
import tflearn
#import matplotlib.pyplot as plot
import tflearn.datasets.mnist as mt

trainX, trainY, testX, testY = mt.load_data(one_hot=True)


tf.reset_default_graph()
    
net=tflearn.input_data([None, 784])
    
net = tflearn.fully_connected(net, 242, activation='relu6')
net = tflearn.fully_connected(net, 542, activation='relu6')
    #net = tflearn.fully_connected(net, 10, activation='Linear')
    
net = tflearn.fully_connected(net, 10, activation='softmax')
net = tflearn.regression(net, optimizer='Adam', learning_rate=0.001, loss='categorical_crossentropy')
    
model = tflearn.DNN(net)
#return model
#model = build_model()
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=196, n_epoch=14)

predictions = np.array(model.predict(testX)).argmax(axis=1)

actual = testY.argmax(axis=1)
acc = np.mean(predictions == actual, axis=0)

print("The accuracy is:", acc)
    