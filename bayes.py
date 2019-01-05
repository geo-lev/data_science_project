#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 03:54:39 2019

@author: geo-lev
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

df_train = pd.read_csv('train.csv')
y_train = df_train[['PAX']]

month = lambda x : datetime.strptime(x , '%Y-%m-%d' ).month
df_train['month'] = df_train['DateOfDeparture'].map(month)

day = lambda x :  datetime.strptime(x , '%Y-%m-%d' ).day
df_train['day'] = df_train['DateOfDeparture'].map(day)


df_train.drop(df_train.columns[[0,1,2,3,5,6,7,11]], axis=1 , inplace = True)



from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(categorical_features = [4,5])
df_train= enc.fit_transform(df_train).toarray()

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
df_train = sc.fit_transform(df_train)
#X_test = sc.transform(X_test)
y_train = np.ravel(y_train)

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(df_train , y_train , test_size = 0.25 )

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import f1_score
x = f1_score(y_test , y_pred , average='micro')