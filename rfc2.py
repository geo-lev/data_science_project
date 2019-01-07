import pandas as pd
import numpy as np
from datetime import datetime
import haversine

def process_df(df): 
    df["distance"] = df_train.apply(lambda row : haversine.haversine((row["LatitudeDeparture"], row["LongitudeDeparture"]),
        (row["LatitudeArrival"], row["LongitudeArrival"])), axis=1)

    year = lambda x : datetime.strptime(x, '%Y-%m-%d').year
    df['year'] = df['DateOfDeparture'].map(year)

    month = lambda x : datetime.strptime(x , '%Y-%m-%d' ).month
    df['month'] = df['DateOfDeparture'].map(month)

    day = lambda x :  datetime.strptime(x , '%Y-%m-%d' ).day
    df['day'] = df['DateOfDeparture'].map(day)

    weekdays = lambda x : datetime.strptime(x , '%Y-%m-%d' ).isoweekday()
    df['weekday'] = df['DateOfDeparture'].map(weekdays)
    
    season = {11: 'Winter', 12: 'Winter', 1: 'Winter', 2: 'Spring', 3: 'Spring', 4: 'Spring', 
              5: 'Summer', 6: 'Summer', 7: 'Summer', 8: 'Autumn', 9: 'Autumn', 10: 'Autumn'}

    df['season'] = df['month'].apply(lambda x : season[x])
    
    return df
    
df_train = pd.read_csv('train.csv')
y_label_train = df_train[['PAX']]
df_train= process_df(df_train)

df_test = pd.read_csv('test.csv')
df_test = process_df(df_test)


df_train.drop(df_train[['DateOfDeparture','CityDeparture','CityArrival','PAX','LongitudeDeparture','distance','std_wtd',
                        'LatitudeDeparture','LongitudeArrival','LatitudeArrival']], axis=1 , inplace = True)
    
df_test.drop(df_test[['DateOfDeparture','CityDeparture','CityArrival','LongitudeDeparture',
                        'LatitudeDeparture','LongitudeArrival','LatitudeArrival','distance','std_wtd',]], axis=1 , inplace = True)

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(df_train , y_label_train , test_size = 0.1 ,random_state=0)
 

from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.compose import ColumnTransformer

coltransf = ColumnTransformer([('one_hot',OneHotEncoder(categories='auto',sparse=False) ,
                                ['Departure','Arrival','day','month','year','weekday','season'])])
#                                ('scaling', MinMaxScaler() ,['WeeksToDeparture'] )])

X_train = coltransf.fit_transform(df_train)
X_test = coltransf.transform(X_test)


df_test = coltransf.fit_transform(df_test)

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline

clf = RandomForestClassifier(n_estimators=512)
clf.fit(X_train,y_label_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import f1_score
x = f1_score(y_test , y_pred , average='micro')