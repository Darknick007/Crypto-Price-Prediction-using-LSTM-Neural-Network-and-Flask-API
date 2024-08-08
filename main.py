import sys
import io

# setting utf 8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.metrics import mean_squared_error
import math
from flask import Flask, request, jsonify
from datetime import timedelta

# Step 1: Data Collection
API_KEY = 'API'  # Inserisci la tua chiave API qui
SYMBOL = 'BTC'
MARKET = 'USD'

url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={SYMBOL}&market={MARKET}&apikey={API_KEY}'

response = requests.get(url)
data = response.json()

# Debug: print the entire response to check for errors or messages
print("API Response:", data)

# Check if the response contains the expected data
if 'Time Series (Digital Currency Daily)' in data:
    # Extract data if the key exists
    df = pd.DataFrame.from_dict(data['Time Series (Digital Currency Daily)'], orient='index')
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)

    # Print the column names to debug
    print("Column names:", df.columns)

    # Keep only relevant columns
    df = df[['1. open', '2. high', '3. low', '4. close', '5. volume']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df.sort_index(inplace=True)

    # Step 2: Data Preprocessing
    # Fill missing values
    df = df.ffill()
    # Normalization
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    # Convert the scaled data back to a DataFrame
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)
else:
    # Handle the case where the key is missing (e.g., print an error message and exit)
    print("Error: 'Time Series (Digital Currency Daily)' key not found in data")
    df_scaled = None

# Proceed only if df_scaled is defined
if df_scaled is not None:
    # Prepare the data for LSTM
    def create_dataset(df, time_step=1):
        dataX, dataY = [], []
        for i in range(len(df) - time_step - 1):
            a = df[i:(i + time_step), :]
            dataX.append(a)
            dataY.append(df[i + time_step, 3])  # Predicting the 'Close' price
        return np.array(dataX), np.array(dataY)

    time_step = 60
    X, y = create_dataset(df_scaled.values, time_step)

    # Split into train and test sets
    train_size = int(len(X) * 0.8)
    test_size = len(X) - train_size
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Step 3: Create the LSTM model
    model = Sequential()
    model.add(Input(shape=(time_step, X_train.shape[2])))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, batch_size=1, epochs=1)

    # Step 4: Model Evaluation
    # Predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse transform predictions
    train_predict = scaler.inverse_transform(
        np.concatenate((train_predict, np.zeros((train_predict.shape[0], df_scaled.shape[1]-1))), axis=1)
    )[:,0]
    test_predict = scaler.inverse_transform(
        np.concatenate((test_predict, np.zeros((test_predict.shape[0], df_scaled.shape[1]-1))), axis=1)
    )[:,0]

    # Calculate RMSE
    train_score = math.sqrt(mean_squared_error(y_train, train_predict))
    test_score = math.sqrt(mean_squared_error(y_test, test_predict))

    print(f'Train RMSE: {train_score}')
    print(f'Test RMSE: {test_score}')

# Step 5: Flask Web Application
    app = Flask(__name__)

    @app.route('/predict', methods=['POST'])
    def predict():
        date = request.json['date']
        date = pd.to_datetime(date)

        # Check if the date is in the future
        if date > df.index[-1]:
            # Calculate how many days ahead the prediction is
            days_ahead = (date - df.index[-1]).days

            # Start with the last known sequence
            input_data = df_scaled.values[-time_step:].reshape(1, time_step, df_scaled.shape[1])

            # Predict iteratively
            for _ in range(days_ahead):
                prediction = model.predict(input_data)
                # Concatenate the prediction with zeros to match the feature dimensions
                prediction = np.concatenate((prediction, np.zeros((1, df_scaled.shape[1] - 1))), axis=1)
                # Append the prediction to the input data and roll the window
                input_data = np.append(input_data[:, 1:, :], prediction.reshape(1, 1, df_scaled.shape[1]), axis=1)

            # Inverse transform the final prediction
            prediction = scaler.inverse_transform(prediction)[:, 0]
        else:
            # Handle past dates as before
            if date not in df.index:
                return jsonify({'error': 'Date not in dataset'}), 400

            idx = df.index.get_loc(date)
            if idx < time_step:
                return jsonify({'error': 'Not enough data to make prediction'}), 400

            input_data = df_scaled.values[idx-time_step:idx].reshape(1, time_step, df_scaled.shape[1])
            prediction = model.predict(input_data)
            prediction = scaler.inverse_transform(
                np.concatenate((prediction, np.zeros((prediction.shape[0], df_scaled.shape[1] - 1))), axis=1)
            )[:, 0]

        return jsonify({'predicted_price': prediction[0]})

    if __name__ == '__main__':
        app.run(debug=True)