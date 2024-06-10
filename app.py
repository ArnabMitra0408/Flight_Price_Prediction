
from flask import Flask, request, render_template
from datetime import datetime, timedelta
import sklearn
import joblib
import pickle
import numpy as np

app = Flask(__name__)
model = joblib.load("XGB_Regression.pkl")

with open('airport_encoder.pkl', 'rb') as f1:
    airport_encoder = pickle.load(f1)


inputs=[]

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict",methods=["POST"])
def predict():
    if request.method=="POST":
        #day of flight
        day_of_flight=request.form["Day_of_Flight"]
        if day_of_flight=='Monday':
            day_of_flight=1
        elif day_of_flight=='Tuesday':
            day_of_flight=2
        elif day_of_flight=='Wednesday':
            day_of_flight=3
        elif day_of_flight=='Thursday':
            day_of_flight=4
        elif day_of_flight=='Friday':
            day_of_flight=5
        elif day_of_flight=="Saturday":
            day_of_flight=6
        else:
            day_of_flight=7
        inputs.append(int(day_of_flight))

        #NumDays
        search_date=request.form["Search_Date"]
        flight_date=request.form["Flight_Date"]
        
        num_day=datetime.strptime(search_date, '%Y-%m-%d').date()-datetime.strptime(flight_date, '%Y-%m-%d').date()
        num_day=num_day.days
        inputs.append(int(num_day))

        #Starting Airport
        starting_airport=airport_encoder.transform([request.form["Starting_Airport"]])[0]
        inputs.append(int(starting_airport))

        #Destination Airport
        destination_airport=airport_encoder.transform([request.form["Destination_Airport"]])[0]
        inputs.append(int(destination_airport))

        #Flight_time_in_minutes
        flight_time=request.form['Flight_Time']
        inputs.append(int(flight_time))


        #isBasicEconomy
        IsBasicEconomy=request.form['IsBasicEconomy']

        if IsBasicEconomy=='N':
            inputs.append(int(0))
        else:
            inputs.append(int(1))

        #isRefundable
        IsRefundable=request.form["IsRefundable"]
        if IsRefundable=='N':
            inputs.append(int(0))
        else:
            inputs.append(int(1))

        #isNonStop
        IsNonStop=request.form["IsNonStop"]
        if IsNonStop=='N':
            inputs.append(int(0))
        else:
            inputs.append(int(1))

        #seatsRemaining
        Seats_Remaining=request.form["Seats_Remaining"]
        inputs.append(int(Seats_Remaining))

        #totalTravelDistance

        Travel_Distance=request.form["Travel_Distance"]
        inputs.append(float(Travel_Distance))

        #Num_Segments
        NumSegments=request.form["NumSegments"]
        inputs.append(int(NumSegments))

        #cabin_code_weight
        cabin_code_weights={'coach':1,'premium coach':2,'business':3,'first':4}
        CabinCode=request.form["CabinCode"]
        segment_c_code=CabinCode.split("||")
        weight=0
        for c_code in segment_c_code:
            weight += cabin_code_weights[c_code]
        inputs.append(weight)
        input=np.array(inputs)
        input=input.reshape(1,-1)
        fare=model.predict(input)[0]
    return render_template("index.html",fare=fare)

if __name__ == "__main__":
    app.run(debug=True)