import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import matplotlib.dates as mdates


#Downloading the S&P 500 from date 01-01-2020 to 01-01-2024
spy = yf.download('SPY', start='2014-01-01', end='2024-01-01')

#Train/Validation split
training_data = spy['Adj Close']['2010-01-01':'2021-12-31']
validation_data = spy['Adj Close']['2022-01-01':'2024-01-01']

#Reshaping to create a 2D array with one column in order to fit it to the model
training_set = training_data.values.reshape(-1, 1)
validation_set = validation_data.values.reshape(-1, 1)

#Rescaling the data in order to be in a range that the model can process them better
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)
validation_set_scaled = sc.transform(validation_set)


#its a function that creates sequences from the data as following
#if data is (3,4,5,6,7) it creates x = (3,4,5,6) and y = (7)
#for the whole dataset in sequences of 60 instances
def create_sequences(data, seq_len=60):
    x = []
    y = []
    for i in range(seq_len, len(data)):
        x.append(data[i-seq_len:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)


#The x_train, y_train and x_validation, y_validation are getting the sequences from the scaled training and validation datasets
X_train, y_train = create_sequences(training_set_scaled)
X_validation, y_validation = create_sequences(validation_set_scaled)

#the datasets are reshaped again in order to be the inpur for the model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_validation = np.reshape(X_validation, (X_validation.shape[0], X_validation.shape[1], 1))

#The model is simple, it consists of 3 LSTM layers, seperated by dropouts + one fully connected layer for output
model = keras.Sequential([
    # LSTM layers
    keras.layers.LSTM(units=200, return_sequences=True),
    keras.layers.Dropout(0.1),
    keras.layers.LSTM(units=200, return_sequences=True),
    keras.layers.Dropout(0.1),
    keras.layers.LSTM(units=200, return_sequences=False),
    keras.layers.Dropout(0.1),

    # Dense layer
    keras.layers.Dense(units=1)
])

#MeanSquaredError is used bacause of the nature of the data with adam optimizer
model.compile(optimizer='adam', loss='mean_squared_error')

#The model is train for 100 epochs
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_validation, y_validation))

#Plot for Training to Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/home/msensis/Documents/Projects/ML/PredictingStockMarcketValues/PredictingStockMarketValues/figs/Loss.png')
plt.show()
plt.clf()

#Load fresh data to prefict as testing
spy_test = yf.download('SPY', start='2024-01-01', end='2024-03-31')

#The data preprocessing in order to get the data on the same track as the training_data
real_stock_price = spy_test['Adj Close'].values.reshape(-1, 1)
dataset_total = pd.concat((spy['Adj Close'], spy_test['Adj Close']), axis=0)
inputs = dataset_total[len(dataset_total) - len(spy_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#Predicting the Stock of S&P 500 for dates it was not trained
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

date_range = pd.date_range(start='2024-01-01', periods=len(predicted_stock_price), freq='B')

#Plotting the real value of stock to predicted
plt.figure(figsize=(10, 6))
plt.plot(spy_test.index, real_stock_price, color='black', label='SPY Stock Price')
plt.plot(date_range, predicted_stock_price, color='green', label='Predicted SPY Stock Price')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.title('SPY Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('SPY Stock Price')
plt.legend()
plt.savefig('/home/msensis/Documents/Projects/ML/PredictingStockMarcketValues/PredictingStockMarketValues/figs/StockPricePrediction.png')
plt.show()