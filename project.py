#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 18:10:08 2019

@author: geo-lev
"""

import pandas as pd
import keras 
import numpy as np

from sklearn.model_selection import train_test_split

df_train = pd.read_csv('train.csv')

df_test = pd.read_csv('test.csv')

y_train = df_train[['PAX']]

#df_train , df_test,y_train , y_test = train_test_split(df_train , y_train , test_size=0.3 , random_state=42)

df_train.drop(df_train.columns[[0,1,2,3,5,6,7,11]] , axis=1 , inplace=True)

df_test.drop(df_test.columns[[0,1,2,3,5,6,7]] , axis=1 , inplace=True)

from sklearn.linear_model import LogisticRegression

X_train = df_train
X_test = df_test
y_train = np.ravel(y_train)

clf = LogisticRegression()

clf.fit(X_train , y_train)

y_pred = clf.predict(X_test)

import csv
with open('y_pred.csv' , 'w' , newline='') as csvfile:
    writer = csv.writer(csvfile , delimiter=',')
    writer.writerow(['Id','Label'])
    for i in range(y_pred.shape[0]):
        writer.writerow([i,y_pred[i]])

from sklearn.metrics import f1_score
print(f1_score(y_test , y_pred , average='micro'))