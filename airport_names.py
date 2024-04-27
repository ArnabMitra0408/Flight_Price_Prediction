
#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyairports.airports import Airports

data=pd.read_csv('Data/Flight_Data.csv')

airports = Airports()
airports.airport_iata('DTW')
starting_airport=set(data['startingAirport'].values)
destination_airport=set(data['destinationAirport'].values)
all_airports=list(set(list(starting_airport) + list(destination_airport)))
names={}
for iata in all_airports:
    airport_info=airports.airport_iata(iata)
    names[iata]=airport_info[0]
data['destination_airport_name'] = data['destinationAirport'].map(names)
data['starting_airport_name']= data['startingAirport'].map(names)

#Rearrange Destination airport
column_index = data.columns.get_loc('destinationAirport') + 1
data.insert(column_index, 'destination_airport_name', data.pop('destination_airport_name'))

#Rearrange Starting airport
column_index = data.columns.get_loc('startingAirport') + 1
data.insert(column_index, 'starting_airport_name', data.pop('starting_airport_name'))

data.to_csv('Data/Flight_Data_with_airport_names.csv',index=False)

data.head()