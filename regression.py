#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 18:10:08 2019

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


df_train.drop(df_train.columns[[0,2,6,11]], axis=1 , inplace = True)

from sklearn.preprocessing import LabelEncoder
lenc = LabelEncoder()
lenc.fit(df_train['Departure'])
df_train['Departure'] = lenc.transform(df_train['Departure'])
df_train['Arrival'] = lenc.transform(df_train['Arrival'])
#lenc.fit(df_train['CityDeparture'])
#df_train['CityDeparture'] = lenc.transform(df_train['CityDeparture'])
#df_train['CityArrival'] = lenc.transform(df_train['CityArrival'])

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(categorical_features = [0,3,8,9])
df_train= enc.fit_transform(df_train).toarray()

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
df_train = sc.fit_transform(df_train)
#X_test = sc.transform(X_test)
y_train = np.ravel(y_train)

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(df_train , y_train , test_size = 0.25 )

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X_train , y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import f1_score
x = f1_score( y_test , y_pred , average='micro')