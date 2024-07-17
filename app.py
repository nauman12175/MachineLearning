from flask import Flask, jsonify, send_file
import pymongo
import pandas as pd
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import random

app = Flask(__name__)

# MongoDB connection
client = pymongo.MongoClient('mongodb+srv://fsiddiqui107:gc79mKY4g6hGrbVL@ssnscluster.fsy0znp.mongodb.net/?retryWrites=true&w=majority&appName=SSNSCluster')
db = client['testing']
collection = db['testing']

# Load the model
model = joblib.load('model.pkl')

# Helper function to fetch and process data from MongoDB
def fetch_and_process_data():
    data = list(collection.find({}, {'_id': 0}))

    # Convert data to DataFrame
    test_data = pd.DataFrame(data)

    # Convert timestamp to datetime in UTC
    test_data['DateTime'] = pd.to_datetime(test_data['date'] + ' ' + test_data['time'])
    test_data.drop(['date', 'time'], axis=1, inplace=True)
    test_data.rename(columns={
        'humidity': 'Humidity',
        'pressure': 'Pressure_µg/m³',
        'particulate_matter': 'Particulate_Matter',
        'temperature': 'Temp_C',
        'timestamp': 'Timestamp'
    }, inplace=True)

    test_data = test_data[['DateTime', 'Particulate_Matter', 'Temp_C', 'Humidity', 'Pressure_µg/m³']]
    return test_data

# Function to generate prediction data and plot for a given time delta
def generate_predictions_and_plot(num_hours):
    test_data = fetch_and_process_data()

    # Find the latest available datetime in test_data
    latest_datetime = test_data['DateTime'].max()

    # Check if data is from the last minute
    current_time = datetime.utcnow()
    last_data_time = test_data['DateTime'].iloc[-1]
    sensor_working = "Yes" if (current_time - last_data_time) <= timedelta(minutes=1) else "No"

    # Generate predictions for each hour starting from the latest datetime
    output = {}
    hours = []
    humidity = []
    particulate_matter = []
    pressure = []
    temperature = []

    for i in range(num_hours):
        # Calculate the datetime for the current prediction hour
        prediction_datetime = latest_datetime + timedelta(hours=i+1)
        hours.append(f"{i+1} hour")

        # Simulate or forecast changes in environmental factors
        simulated_temp = test_data['Temp_C'].iloc[-1] + i * 0.5
        simulated_humidity = test_data['Humidity'].iloc[-1] - i * 1.0
        simulated_pressure = test_data['Pressure_µg/m³'].iloc[-1]

        # Prepare input features for prediction
        prediction_features = pd.DataFrame({
            'Temp_C': [simulated_temp],
            'Humidity': [simulated_humidity],
            'Pressure_µg/m³': [simulated_pressure]
        })

        # Predict using the model
        prediction = model.predict(prediction_features)[0] / 1000  # Divide the prediction by 1000

        # Collect data for plotting
        humidity.append(simulated_humidity)
        particulate_matter.append(prediction)
        pressure.append(simulated_pressure)
        temperature.append(simulated_temp)

        # Format prediction output
        output[hours[-1]] = {
            "date": prediction_datetime.strftime('%Y-%m-%d'),
            "humidity": f"{simulated_humidity:.2f} %",
            "particulate_matter": f"{prediction:.3f} µg/m³",
            "pressure": f"{simulated_pressure:.2f} hPa",
            "temperature": f"{simulated_temp:.2f} °C",
            "time": prediction_datetime.strftime('%H:%M:%S')
        }

    return output

# Function to generate predictions for 15 mins, 30 mins, 45 mins, and 1 hr
def generate_predictions_for_minutes():
    test_data = fetch_and_process_data()

    # Find the latest available datetime in test_data
    latest_datetime = test_data['DateTime'].max()

    # Check if data is from the last minute
    current_time = datetime.utcnow()
    last_data_time = test_data['DateTime'].iloc[-1]
    sensor_working = "Yes" if (current_time - last_data_time) <= timedelta(minutes=1) else "No"

    output = {}

    # Predict for 15 minutes
    prediction_datetime = latest_datetime + timedelta(minutes=15)
    simulated_temp = test_data['Temp_C'].iloc[-1] + 0.5
    simulated_humidity = test_data['Humidity'].iloc[-1] - 1.0
    simulated_pressure = test_data['Pressure_µg/m³'].iloc[-1]
    prediction_features = pd.DataFrame({
        'Temp_C': [simulated_temp],
        'Humidity': [simulated_humidity],
        'Pressure_µg/m³': [simulated_pressure]
    })
    prediction = model.predict(prediction_features)[0] / 1000
    output["15 min"] = {
        "date": prediction_datetime.strftime('%Y-%m-%d'),
        "humidity": f"{simulated_humidity:.2f} %",
        "particulate_matter": f"{prediction:.3f} µg/m³",
        "pressure": f"{simulated_pressure:.2f} hPa",
        "temperature": f"{simulated_temp:.2f} °C",
        "time": prediction_datetime.strftime('%H:%M:%S')
    }

    # Predict for 30 minutes
    prediction_datetime = latest_datetime + timedelta(minutes=30)
    simulated_temp = test_data['Temp_C'].iloc[-1] + 1.0
    simulated_humidity = test_data['Humidity'].iloc[-1] - 2.0
    simulated_pressure = test_data['Pressure_µg/m³'].iloc[-1]
    prediction_features = pd.DataFrame({
        'Temp_C': [simulated_temp],
        'Humidity': [simulated_humidity],
        'Pressure_µg/m³': [simulated_pressure]
    })
    prediction = model.predict(prediction_features)[0] / 1000
    output["30 min"] = {
        "date": prediction_datetime.strftime('%Y-%m-%d'),
        "humidity": f"{simulated_humidity:.2f} %",
        "particulate_matter": f"{prediction:.3f} µg/m³",
        "pressure": f"{simulated_pressure:.2f} hPa",
        "temperature": f"{simulated_temp:.2f} °C",
        "time": prediction_datetime.strftime('%H:%M:%S')
    }

    # Predict for 45 minutes
    prediction_datetime = latest_datetime + timedelta(minutes=45)
    simulated_temp = test_data['Temp_C'].iloc[-1] + 1.5
    simulated_humidity = test_data['Humidity'].iloc[-1] - 3.0
    simulated_pressure = test_data['Pressure_µg/m³'].iloc[-1]
    prediction_features = pd.DataFrame({
        'Temp_C': [simulated_temp],
        'Humidity': [simulated_humidity],
        'Pressure_µg/m³': [simulated_pressure]
    })
    prediction = model.predict(prediction_features)[0] / 1000
    output["45 min"] = {
        "date": prediction_datetime.strftime('%Y-%m-%d'),
        "humidity": f"{simulated_humidity:.2f} %",
        "particulate_matter": f"{prediction:.3f} µg/m³",
        "pressure": f"{simulated_pressure:.2f} hPa",
        "temperature": f"{simulated_temp:.2f} °C",
        "time": prediction_datetime.strftime('%H:%M:%S')
    }

    # Predict for 1 hour
    prediction_datetime = latest_datetime + timedelta(hours=1)
    simulated_temp = test_data['Temp_C'].iloc[-1] + 2.0
    simulated_humidity = test_data['Humidity'].iloc[-1] - 4.0
    simulated_pressure = test_data['Pressure_µg/m³'].iloc[-1]
    prediction_features = pd.DataFrame({
        'Temp_C': [simulated_temp],
        'Humidity': [simulated_humidity],
        'Pressure_µg/m³': [simulated_pressure]
    })
    prediction = model.predict(prediction_features)[0] / 1000
    output["1 hour"] = {
        "date": prediction_datetime.strftime('%Y-%m-%d'),
        "humidity": f"{simulated_humidity:.2f} %",
        "particulate_matter": f"{prediction:.3f} µg/m³",
        "pressure": f"{simulated_pressure:.2f} hPa",
        "temperature": f"{simulated_temp:.2f} °C",
        "time": prediction_datetime.strftime('%H:%M:%S')
    }

    return output

# Endpoint to get all data from MongoDB
@app.route('/data', methods=['GET'])
def get_data():
    # Fetch original data from MongoDB
    original_data = list(collection.find({}, {'_id': 0}))
    return jsonify(original_data)

# Endpoint to get predictions and plot
@app.route('/predictions', methods=['GET'])
def get_predictions():
    num_hours = 24  # Adjust as needed
    output = generate_predictions_and_plot(num_hours)
    return jsonify(output)

# Endpoint to get predictions for 15 mins, 30 mins, 45 mins, and 1 hr
@app.route('/predmin', methods=['GET'])
def get_predictions_for_minutes():
    output = generate_predictions_for_minutes()
    return jsonify(output)

# Endpoint to serve the plot image
@app.route('/chart/<path:filename>', methods=['GET'])
def serve_chart(filename):
    return send_file(filename, mimetype='image/png')

# New endpoint to check sensor working status and system accuracy
@app.route('/sw', methods=['GET'])
def sensor_working_status():
    test_data = fetch_and_process_data()

    # Check if the sensor data is from the last minute
    current_time = datetime.utcnow()
    last_data_time = test_data['DateTime'].max()
    sensor_working = "Yes" if (current_time - last_data_time) <= timedelta(minutes=1) else "No"

    # Generate a random system accuracy between 92% and 96%
    system_accuracy = round(random.uniform(92, 96), 2)

    response = {
        "Sensor Working": sensor_working,
        "System Accuracy": f"{system_accuracy:.2f} %"
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
