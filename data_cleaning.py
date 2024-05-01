#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyairports.airports import Airports
from datetime import datetime
from geopy.distance import geodesic

def import_data():
    data=pd.read_csv('Data/Flight_Data.csv',usecols=['searchDate', 'flightDate', 'startingAirport', 'destinationAirport', 
       'travelDuration', 'isBasicEconomy', 'isRefundable', 'isNonStop',
       'seatsRemaining', 'totalTravelDistance',
       'segmentsArrivalAirportCode', 'segmentsDepartureAirportCode',
       'segmentsAirlineName',
       'segmentsEquipmentDescription','segmentsCabinCode','totalFare'])
    return data

def get_airport_names(data):
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
    return data

def calculate_distance(starting_coord,destination_coord):
    return geodesic(starting_coord, destination_coord).miles

def Travel_Distance_Nulls(data):
    for flight in range(data.shape[0]):
        if pd.isnull(data.at[flight, 'totalTravelDistance']):
            airports = Airports()
            starting_airport=airports.airport_iata(data['startingAirport'][flight])
            starting_lat=float(starting_airport[5])
            starting_lon=float(starting_airport[6])

            destination_airport=airports.airport_iata(data['destinationAirport'][flight])
            destination_lat=float(destination_airport[5])
            destination_lon=float(destination_airport[6])

            total_distance=calculate_distance((starting_lat,starting_lon),(destination_lat,destination_lon))
            data.loc[flight,'totalTravelDistance']=round(total_distance,2)
    return data

if __name__ =='__main__':

    print("--------------Importing Data------------------")

    data=import_data()

    print("--------------Fetching Airport Names------------------")

    data=get_airport_names(data)

    print('------------------------Handling Null Values in TravelDistance---------')

    data=Travel_Distance_Nulls(data)
    
    print("--------------Saving Cleaned Data in CSV------------------")

    data.to_csv("Data/Flight_Data_Cleaned.csv",index=False)



