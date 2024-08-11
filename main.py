import sys
import io
import os
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention, Input, Layer
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from flask import Flask, request, jsonify
import logging
import math

# Disable TensorFlow OneDNN optimizations if needed
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set console output to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configure logging
logging.basicConfig(level=logging.INFO)

# API configuration
API_KEY = ''  # Replace with your Alpha Vantage API key
SYMBOL = 'BTC'
MARKET = 'USD'

url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={SYMBOL}&market={MARKET}&apikey={API_KEY}'

# Step 1: Data Collection
try:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    logging.info("API Response: %s", data)
except requests.exceptions.RequestException as e:
    logging.error(f"Request failed: {e}")
    data = None

if data and 'Time Series (Digital Currency Daily)' in data:
    df = pd.DataFrame.from_dict(data['Time Series (Digital Currency Daily)'], orient='index')
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)

    # Keep only relevant columns
    df = df[['1. open', '2. high', '3. low', '4. close', '5. volume']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df.sort_index(inplace=True)

    # Step 2: Feature Engineering
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # Manually calculate RSI
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Volatility
    df['Volatility'] = df['Close'].rolling(window=10).std()

    # Manually calculate MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26

    # Manually calculate Bollinger Bands
    df['Bollinger_High'] = df['Close'].rolling(window=20).mean() + df['Close'].rolling(window=20).std() * 2
    df['Bollinger_Low'] = df['Close'].rolling(window=20).mean() - df['Close'].rolling(window=20).std() * 2

    # Drop any NaN values resulting from rolling calculations
    df.dropna(inplace=True)

    # Data Preprocessing
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler = MinMaxScaler(feature_range=(0, 1))

    # Scale the 'Close' price separately to avoid data leakage
    df['Close'] = target_scaler.fit_transform(df[['Close']])
    df[df.columns.difference(['Close'])] = feature_scaler.fit_transform(df[df.columns.difference(['Close'])])
    
    df_scaled = df

else:
    logging.error("'Time Series (Digital Currency Daily)' not found in data")
    df_scaled = None

# Proceed only if df_scaled is defined
if df_scaled is not None:

    def create_dataset(df, time_step=1):
        dataX, dataY = [], []
        for i in range(len(df) - time_step):
            a = df[i:(i + time_step), :]
            dataX.append(a)
            dataY.append(df[i + time_step, 3])  # Predict the 'Close' price
        return np.array(dataX), np.array(dataY)

    time_step = 60
    X, y = create_dataset(df_scaled.values, time_step)

    # Split into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Further split training set into training and validation sets
    val_size = int(len(X_train) * 0.2)
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]

    # Step 3: Custom Layer for Summing over Sequence Dimension
    class SumLayer(Layer):
        def call(self, inputs):
            return tf.reduce_sum(inputs, axis=1)

    # Step 4: Define Model Without Keras Tuner
    def build_model():
        inputs = Input(shape=(time_step, X_train.shape[2]))
        x = LSTM(units=100, return_sequences=True)(inputs)

        # Adding Attention layer
        attention = Attention()([x, x])
        x = SumLayer()(attention)  # Summing over the sequence dimension

        x = Dense(50, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)

        # Compile the model with a lower learning rate for better convergence
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
        return model

    model = build_model()

    # Train the model with early stopping
    model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_val, y_val),
              callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)])

    # Step 5: Model Evaluation
    def evaluate_model(model, X_train, y_train, X_test, y_test, target_scaler):
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Inverse transform predictions
        train_predict_inv = target_scaler.inverse_transform(train_predict)
        test_predict_inv = target_scaler.inverse_transform(test_predict)

        y_train_inv = target_scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1))

        # Calculate RMSE
        train_score = math.sqrt(mean_squared_error(y_train_inv, train_predict_inv))
        test_score = math.sqrt(mean_squared_error(y_test_inv, test_predict_inv))

        logging.info(f'Train RMSE: {train_score}')
        logging.info(f'Test RMSE: {test_score}')

        return train_score, test_score

    # Call the evaluation function
    train_score, test_score = evaluate_model(model, X_train, y_train, X_test, y_test, target_scaler)

    # Step 6: Flask Web Application
    app = Flask(__name__)

    def send_notification(message):
        # Replace this with your preferred notification method (e.g., email, SMS, push notification)
        logging.info(f"Notification: {message}")  # For now, simply log the message

    @app.route('/predict', methods=['POST'])
    def predict():
        date = request.json['date']
        date = pd.to_datetime(date)

        if date > df.index[-1]:
            days_ahead = (date - df.index[-1]).days
            input_data = df_scaled.values[-time_step:].reshape(1, time_step, df_scaled.shape[1])

            for _ in range(days_ahead):
                prediction = model.predict(input_data)
                prediction = np.concatenate((prediction, np.zeros((1, df_scaled.shape[1] - 1))), axis=1)
                input_data = np.append(input_data[:, 1:, :], prediction.reshape(1, 1, df_scaled.shape[1]), axis=1)

            prediction = target_scaler.inverse_transform(prediction)[:, 0]
            threshold = 50000  # Adjust the threshold as needed
            if prediction[0] > threshold:
                send_notification(f"The prediction for {date} exceeds the threshold: {prediction[0]}")
        else:
            if date not in df.index:
                return jsonify({'error': 'Date not in dataset'}), 400

            idx = df.index.get_loc(date)
            if idx < time_step:
                return jsonify({'error': 'Not enough data to make prediction'}), 400

            input_data = df_scaled.values[idx-time_step:idx].reshape(1, time_step, df_scaled.shape[1])
            prediction = model.predict(input_data)
            prediction = target_scaler.inverse_transform(
                np.concatenate((prediction, np.zeros((prediction.shape[0], df_scaled.shape[1] - 1))), axis=1)
            )[:, 0]

        return jsonify({'predicted_price': prediction[0]})

    # Step 7: Save and Load the Model
    def save_model(model, file_name='btc_lstm_model.h5'):
        model.save(file_name)
        logging.info(f"Model saved to {file_name}")

    def load_model(file_name='btc_lstm_model.h5'):
        if os.path.exists(file_name):
            model = tf.keras.models.load_model(file_name)
            logging.info(f"Model loaded from {file_name}")
            return model
        else:
            logging.error(f"Model file {file_name} not found")
            return None

    if __name__ == '__main__':
        # Save the trained model for later use
        save_model(model)

        # To load an existing model, use:
        # model = load_model()

        app.run(debug=True)
