#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 23:17:29 2019

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


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(categorical_features = [0,3,8,9] , sparse=False)
df_train= enc.fit_transform(df_train)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df_train = sc.fit_transform(df_train)

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(df_train , y_train , test_size = 0.25 )

from imblearn.over_sampling import SMOTE
ovs = SMOTE()
X_train_res , y_train_res = ovs.fit_sample(X_train , y_train)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=500 , random_state = 12)
clf.fit(X_train_res,y_train_res)

y_pred = clf.predict(X_test)

from sklearn.metrics import f1_score
x = f1_score(y_test , y_pred , average='micro')