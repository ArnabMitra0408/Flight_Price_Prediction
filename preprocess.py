#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyairports.airports import Airports
from datetime import datetime

# Importing Dataset
data=pd.read_csv('Data/Flight_Data_New.csv',usecols=['searchDate', 'flightDate', 'startingAirport',
       'starting_airport_name', 'destinationAirport',
       'destination_airport_name', 'fareBasisCode', 
       'travelDuration', 'isBasicEconomy', 'isRefundable', 'isNonStop',
       'seatsRemaining', 'totalTravelDistance', 'segmentsDepartureTimeRaw',
       'segmentsArrivalTimeRaw',
       'segmentsArrivalAirportCode', 'segmentsDepartureAirportCode',
       'segmentsAirlineName', 'segmentsAirlineCode',
       'segmentsEquipmentDescription', 'segmentsDurationInSeconds',
       'segmentsDistance', 'segmentsCabinCode','totalFare'],nrows=1000)

# Extracting Number of days between flight Date and search date
num_days=[]
for dt in range(data.shape[0]):
    num_day=datetime.strptime(data['flightDate'][dt], '%Y-%m-%d').date()-datetime.strptime(data['searchDate'][dt], '%Y-%m-%d').date()
    num_day=num_day.days
    num_days.append(num_day)
data['NumDays']=num_days

column_index = data.columns.get_loc('flightDate') + 1
data.insert(column_index, 'NumDays', data.pop('NumDays'))



# Flight Day (M,T,W,etc)

data['flightDate'] = pd.to_datetime(data['flightDate'])
data['Day_of_Flight'] = data['flightDate'].dt.day_name()  


column_index = data.columns.get_loc('flightDate') + 1
data.insert(column_index, 'Day_of_Flight', data.pop('Day_of_Flight'))

days={'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6, 'Sunday':7}
data['Day_of_Flight']=data['Day_of_Flight'].map(days)


#Flight Time in Minutes
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

data['Flight_time_in_minutes'] = data['travelDuration'].apply(calculate_total_minutes)

column_index = data.columns.get_loc('travelDuration') + 1
data.insert(column_index, 'Flight_time_in_minutes', data.pop('Flight_time_in_minutes'))

data.drop(['travelDuration'],axis=1,inplace=True) 