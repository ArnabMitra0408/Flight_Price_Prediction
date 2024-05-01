#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyairports.airports import Airports
from datetime import datetime
import plotly.express as px
from datetime import datetime, timedelta
import plotly.graph_objects as go
from geopy.distance import geodesic
import pickle

def import_data():
    data=pd.read_csv("Data/Flight_Data_Cleaned.csv")
    return data


def encoding_airports(data):
    from sklearn.preprocessing import LabelEncoder
    airport_encoder=LabelEncoder()
    airport_encoder.fit(data['startingAirport'])
    data['startingAirport']=airport_encoder.transform(data['startingAirport'])
    data['destinationAirport']=airport_encoder.transform(data['destinationAirport'])


    with open('airport_encoder.pkl', 'wb') as f:
        pickle.dump(airport_encoder, f)

    return data,airport_encoder

def Num_Days(data):
    num_days=[]
    for dt in range(data.shape[0]):
        num_day=datetime.strptime(data['flightDate'][dt], '%Y-%m-%d').date()-datetime.strptime(data['searchDate'][dt], '%Y-%m-%d').date()
        num_day=num_day.days
        num_days.append(num_day)
    data['NumDays']=num_days

    column_index = data.columns.get_loc('flightDate') + 1
    data.insert(column_index, 'NumDays', data.pop('NumDays'))
    return data

def day_of_flight(data):
    data['flightDate'] = pd.to_datetime(data['flightDate'])
    data['Day_of_Flight'] = data['flightDate'].dt.day_name()  


    column_index = data.columns.get_loc('flightDate') + 1
    data.insert(column_index, 'Day_of_Flight', data.pop('Day_of_Flight'))

    days={'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6, 'Sunday':7}
    data['Day_of_Flight']=data['Day_of_Flight'].map(days)
    return data

def calculate_total_minutes(duration):
    hour=0
    min=0
    try: 
        hour=int(duration.split('H')[0][-1])
        try:
            min= int(duration.split('H')[1][:-1])
        except:
            min=0
    except:
        try:
            min= int(duration.split('H')[1][:-1])
        except:
            min=0
    return (hour*60)+min


def flight_time_in_minutes(data):

    data['Flight_time_in_minutes'] = data['travelDuration'].apply(calculate_total_minutes)

    column_index = data.columns.get_loc('travelDuration') + 1
    data.insert(column_index, 'Flight_time_in_minutes', data.pop('Flight_time_in_minutes'))

    data.drop(['travelDuration'],axis=1,inplace=True) 
    return data


def boolean_encoding(data):
    t_f={True:1,False:0}
    data['isBasicEconomy']=data['isBasicEconomy'].map(t_f)
    data['isRefundable']=data['isRefundable'].map(t_f)
    data['isNonStop']=data['isNonStop'].map(t_f)
    return data

def num_segments(data):
    num_segments=[]
    for flight in range(data.shape[0]):
        num_segments.append(data['segmentsArrivalAirportCode'][flight].count('|')+1)


    data['Num_Segments']=num_segments
    column_index = data.columns.get_loc('segmentsArrivalAirportCode') + 1
    data.insert(column_index, 'Num_Segments', data.pop('Num_Segments'))
    return data

def cabin_code_weight(data):
    cabin_code_weights={'coach':1,'premium coach':2,'business':3,'first':4}
    weighted_cabin_code=[]

    for cabin_code in range(data.shape[0]):
        weight=0
        segment_c_code=data['segmentsCabinCode'][cabin_code].split("||")
        for c_code in segment_c_code:
            weight += cabin_code_weights[c_code]
        weighted_cabin_code.append(weight)

    data['cabin_code_weight']=weighted_cabin_code
    column_index = data.columns.get_loc('segmentsCabinCode') + 1
    data.insert(column_index, 'cabin_code_weight', data.pop('cabin_code_weight'))
    return data





if __name__ == '__main__':

    data=import_data()

    print('------------------------Encoding Airports-------------------')

    data,airport_encoder=encoding_airports(data)

    print("--------------Extracting Num Days between flight Data and Search Date------------------")

    data=Num_Days(data)

    print("--------------Extracting Flight Day (M,T,etc)------------------")

    data=day_of_flight(data)

    print("--------------Calculating Flight Time in Minutes------------------")

    data=flight_time_in_minutes(data)

    print("--------------Encoding Boolean Features (isBasicEconomy, isRefundable, isNonStop)------------------")

    data=boolean_encoding(data)

    print("--------------Calculating Number of Segments------------------")

    data=num_segments(data)

    print("--------------Assigned Weights to Cabin Codes------------------")

    data=cabin_code_weight(data)

    print('------Saving Data---------')
    data=data[['Day_of_Flight','NumDays','startingAirport','destinationAirport','Flight_time_in_minutes','isBasicEconomy','isRefundable','isNonStop','totalFare','seatsRemaining','totalTravelDistance','Num_Segments','cabin_code_weight']]
    data.to_csv('Data/Flight_Data_Processed.csv',index=False)
