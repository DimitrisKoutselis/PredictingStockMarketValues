import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import matplotlib.dates as mdates

spy = yf.download('SPY', start='2010-01-01', end='2024-01-01')

training_data = spy['Adj Close']['2010-01-01':'2021-12-31']
validation_data = spy['Adj Close']['2022-01-01':'2024-01-01']

training_set = training_data.values.reshape(-1, 1)
validation_set = validation_data.values.reshape(-1, 1)

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)
validation_set_scaled = sc.transform(validation_set)

def create_sequences(data, seq_len=60):
    x = []
    y = []
    for i in range(seq_len, len(data)):  # Corrected
        x.append(data[i-seq_len:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

X_train, y_train = create_sequences(training_set_scaled)
X_validation, y_validation = create_sequences(validation_set_scaled)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_validation = np.reshape(X_validation, (X_validation.shape[0], X_validation.shape[1], 1))

model = keras.Sequential([
    # LSTM layers
    keras.layers.LSTM(units=150, return_sequences=True),
    keras.layers.Dropout(0.1),
    keras.layers.LSTM(units=150, return_sequences=True),
    keras.layers.Dropout(0.1),
    keras.layers.LSTM(units=150, return_sequences=False),
    keras.layers.Dropout(0.1),

    # Dense layer
    keras.layers.Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_validation, y_validation))

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('figs/Loss.png')
plt.show()
plt.clf()

spy_test = yf.download('SPY', start='2024-01-01', end='2024-03-31')

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

predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

date_range = pd.date_range(start='2024-01-01', periods=len(predicted_stock_price), freq='B')


plt.figure(figsize=(10, 6))
plt.plot(spy_test.index, real_stock_price, color='black', label='SPY Stock Price')
plt.plot(date_range, predicted_stock_price, color='green', label='Predicted SPY Stock Price')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.title('SPY Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('SPY Stock Price')
plt.legend()
plt.savefig('figs/StockPricePrediction.png')
plt.show()

#demo