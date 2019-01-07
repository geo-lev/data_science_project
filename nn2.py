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
y_train = df_train[['PAX']]
df_train= process_df(df_train)

df_test = pd.read_csv('test.csv')
df_test = process_df(df_test)


df_train.drop(df_train[['DateOfDeparture','CityDeparture','CityArrival','PAX','LongitudeDeparture','distance','std_wtd',
                        'LatitudeDeparture','LongitudeArrival','LatitudeArrival']], axis=1 , inplace = True)
    
df_test.drop(df_test[['DateOfDeparture','CityDeparture','CityArrival','LongitudeDeparture',
                        'LatitudeDeparture','LongitudeArrival','LatitudeArrival','distance','std_wtd',]], axis=1 , inplace = True)


from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.compose import ColumnTransformer

coltransf = ColumnTransformer([('one_hot',OneHotEncoder(categories='auto',sparse=False) ,
                                ['Departure','Arrival','day','month','year','weekday','season']),
                                ('scaling', MinMaxScaler() ,['WeeksToDeparture'] )])

df_train = coltransf.fit_transform(df_train)
df_test = coltransf.transform(df_test)

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(df_train , y_train , test_size = 0.25 ,random_state=42)
y_train = np.ravel(y_train)



from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

clf = MLPClassifier(alpha=0.55 , hidden_layer_sizes=(64,32,16),random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#import csv 
#with open('y_pred.csv', 'w', newline ='') as csvfile:
#    writer = csv.writer(csvfile, delimiter=',')
#    writer.writerow(['Id', 'Label'])
#    for i in range(y_pred.shape[0]):
#        writer.writerow([i, y_pred[i]])

from sklearn.metrics import f1_score
score = f1_score(y_test , y_pred , average='micro')