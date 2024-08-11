# Crypto-Price-Prediction-using-LSTM-Neural-Network-and-Flask-API

**Crypto Price Prediction with LSTM Neural Network**

This project leverages a machine learning model to predict Bitcoin prices using a Long Short-Term Memory (LSTM) neural network, created with Python and the TensorFlow and Keras libraries. The model is trained on historical Bitcoin data, incorporating various technical indicators to enhance prediction accuracy. An API endpoint is also set up using Flask, allowing users to make predictions based on specified dates.

### Key Features

- **Data Collection**: A script fetches daily Bitcoin price data from the Alpha Vantage API.
  
- **Feature Engineering**: Includes indicators like, Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), Bollinger Bands, and volatility.

- **Model Architecture**: The core is an LSTM network with an in-built attention mechanism, applied to relevant parts of the sequence. A custom TensorFlow layer sums over this sequence dimension.

- **Model Training**: The model employs early stopping to avoid overfitting, with data split into training, validation, and test sets.

- **Evaluation**: Model performance is evaluated using Root Mean Squared Error (RMSE) on both training and testing datasets.

- **Flask Web API**: A Flask web application allows users to input dates and receive alerts if the forecasted price surpasses a given threshold.

- **Model Persistence**: The trained model can be saved to disk and reloaded for future use.

### Requirements

- Python 3.x
- TensorFlow
- NumPy
- Flask
- Requests
- Scikit-learn
- Alpha Vantage API Key (Free tier available)

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Darknick007/Crypto-Price-Prediction-using-LSTM-Neural-Network-and-Flask-API.git
   cd Crypto-Price-Prediction-using-LSTM-Neural-Network-and-Flask-API.git
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the Alpha Vantage API Key**:
   Replace the `API_KEY` placeholder in the script with your Alpha Vantage API key.

### Usage

1. **Setup and Installation**:
   - Clone the repository and install the necessary Python libraries.

2. **Running the Script**:
   - Train the model by running:
     ```bash
     python main.py
     ```
   - The trained model will be saved as `btc_lstm_model.h5`.

   - By default, the API runs on `http://127.0.0.1:5000/`.

3. **Making Predictions**:
   - Post to the `/predict` endpoint with a specific date:
     ```json
     {
       "date": "YYYY-MM-DD"
     }
     ```
   - The API will return a JSON object with the predicted price:
     ```json
     {
       "predicted_price": "price"
     }
     ```

### Model and API Customization

- **Adjust Prediction Threshold**: Modify the threshold in the `predict()` method to customize the notification system.
  
- **Expand Features**: Add more technical indicators or modify existing ones in the feature engineering section.

- **Save and Load Models**: Use the `save_model()` and `load_model()` functions to manage model persistence.

### Caveat

This project is for educational and experimental purposes only and should not be used for real investment or trading decisions. The predictions are based on historical data, and the methods employed may not accurately predict future price movements. Use at your own risk. (used LLM for developing and the description)

### Contributing

Feel free to fork this repository, make pull requests, or open issues for bugs, new features, or questions.

### License

Distributed under the MIT License. See the LICENSE file for more information.

### Contact

For inquiries, you can reach me at `nick007sbt@gmail.com` or create an issue in this repository.
