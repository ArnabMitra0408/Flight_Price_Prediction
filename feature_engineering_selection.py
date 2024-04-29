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

# Importing Dataset
data=pd.read_csv("Data/Flight_Data_Cleaned.csv",usecols=['Day_of_Flight', 'NumDays',
       'startingAirport','destinationAirport','Flight_time_in_minutes', 'isBasicEconomy',
       'isRefundable', 'isNonStop', 'totalFare', 'seatsRemaining',
       'totalTravelDistance','Num_Segments','cabin_code_weight'])

#Encoding Airport Code
print('------------------------Encoding Airports-------------------')

from sklearn.preprocessing import LabelEncoder
airport_encoder=LabelEncoder()
airport_encoder.fit(data['startingAirport'])
data['startingAirport']=airport_encoder.transform(data['startingAirport'])
data['destinationAirport']=airport_encoder.transform(data['destinationAirport'])

print('----Saving airport label encoder-------')

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(airport_encoder, f)

print('------------------------Handling Null Values in TravelDistance---------')

def calculate_distance(starting_coord,destination_coord):
    return geodesic(starting_coord, destination_coord).miles

for flight in range(data.shape[0]):
    if pd.isnull(data.at[flight, 'totalTravelDistance']):
        airports = Airports()
        starting_airport=airports.airport_iata(list(airport_encoder.inverse_transform([data['startingAirport'][flight]]))[0])
        starting_lat=float(starting_airport[5])
        starting_lon=float(starting_airport[6])

        destination_airport=airports.airport_iata(list(airport_encoder.inverse_transform([data['destinationAirport'][flight]]))[0])
        destination_lat=float(destination_airport[5])
        destination_lon=float(destination_airport[6])

        total_distance=calculate_distance((starting_lat,starting_lon),(destination_lat,destination_lon))
        data.loc[flight,'totalTravelDistance']=round(total_distance,2)
        

print('------Saving Data---------')
data.to_csv('Data/Flight_Data_Processed.csv',index=False)
