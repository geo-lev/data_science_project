#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 17:23:03 2019

@author: geo-lev
"""

import pandas as pd
import numpy as np

df_train = pd.read_csv('train.csv')
y_train = df_train[['PAX']]

df_train.drop(df_train.columns[[0,1,2,3,5,6,7,11]], axis=1 , inplace = True)

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(df_train , y_train , test_size = 0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#from keras.models import Sequential
#from keras.layers import Dense

#clf = Sequential()

#clf.add(Dense(output_dim = 6 , init = 'uniform' , activation='linear' , input_dim = 4 ))

#clf.add(Dense(output_dim = 6 , init = 'uniform' , activation = 'linear'))

#clf.add(Dense(output_dim = 1 , init = 'uniform' , activation = 'linear'))

#clf.compile(optimizer='adam' , loss = 'mean_absolute_error', metrics=['accuracy'])

#clf.fit(X_train , y_train,batch_size=256 , nb_epoch = 50)

#y_pred = clf.predict(X_test)

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier()

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import f1_score
x = f1_score(y_test , y_pred , average='micro')