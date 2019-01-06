#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 16:18:46 2019

@author: geo-lev
"""

import pandas as pd
import numpy as np
from datetime import datetime
from imblearn.pipeline import Pipeline
import haversine

df_train = pd.read_csv('train.csv')
y_train = df_train[['PAX']]

df_train["distance"] = df_train.apply(lambda row : haversine.haversine((row["LatitudeDeparture"], row["LongitudeDeparture"]),
        (row["LatitudeArrival"], row["LongitudeArrival"])), axis=1)


year = lambda x : datetime.strptime(x, '%Y-%m-%d').year
df_train['year'] = df_train['DateOfDeparture'].map(year)

month = lambda x : datetime.strptime(x , '%Y-%m-%d' ).month
df_train['month'] = df_train['DateOfDeparture'].map(month)

day = lambda x :  datetime.strptime(x , '%Y-%m-%d' ).day
df_train['day'] = df_train['DateOfDeparture'].map(day)

season = {
    11: 'Winter',
    12: 'Winter',
    1: 'Winter',
    2: 'Spring',
    3: 'Spring',
    4: 'Spring',
    5: 'Summer',
    6: 'Summer',
    7: 'Summer',
    8: 'Autumn',
    9: 'Autumn',
    10: 'Autumn'
}

df_train['season'] = df_train['month'].apply(lambda x : season[x])

df_train.drop(df_train[['DateOfDeparture','CityDeparture','CityArrival','PAX']], axis=1 , inplace = True)

cols_to_encode = ['Departure','Arrival','month','day','year','season']
cols_to_scale = ['WeeksToDeparture','std_wtd','distance']

from sklearn.preprocessing import OneHotEncoder,MinMaxScaler

scaler = MinMaxScaler()
ohe = OneHotEncoder(categories='auto',sparse=False)

scaled_cols = scaler.fit_transform(df_train[cols_to_scale])
encoded_cols = ohe.fit_transform(df_train[cols_to_encode])

processed_df = np.concatenate([scaled_cols , encoded_cols], axis =1)

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(processed_df , y_train , test_size = 0.25 )
y_train = np.ravel(y_train) 

from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier

pipeline = Pipeline([('ovs',SMOTE()) , ('clf',KNeighborsClassifier())])
pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)

from sklearn.metrics import f1_score
score = f1_score(y_test , y_pred , average='micro')