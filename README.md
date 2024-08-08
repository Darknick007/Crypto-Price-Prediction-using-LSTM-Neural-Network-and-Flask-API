# Crypto-Price-Prediction-using-LSTM-Neural-Network-and-Flask-API
This project tries to predict cryptocurrency prices using an LSTM neural network and Flask API. It fetches daily data from Alpha Vantage, processes it for model training, and provides predictions via a web interface. The code is adaptable for different cryptocurrencies and markets by changing the symbol and market parameters.

To use it: -Obtain a free API key from Alpha Vantage. -Integrate the API key into your code. -Customize the symbol (e.g., BTC, ETH) by modifying the symbol variable. Similarly, adjust the market variable as needed. -Once you've set everything up, run main.py. -Make a POST request to your Flask application (I use Postman) with the desired date .

DISCLAIMER

**obviously this project does not predict the market 100%, in fact it is not recommended to use for financial reasons because it follows an LSTM neural network and therefore most of the time it will give non-useful results.
it is an experiment and therefore it is recommended to use for other projects or experiments.**