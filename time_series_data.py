import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

data = np.sin(np.arange(0, 100, 0.1)).reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

time_step = 10
x, y = [], []
for i in range(time_step, len(scaled_data)):
    x.append(scaled_data[i - time_step:i, 0])
    y.append(scaled_data[i, 0])

x, y = np.array(x), np.array(y)
x = x.reshape(x.shape[0], x.shape[1], 1)

train_size = int(len(x) * 0.8)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(time_step,1)))
model.add(Dropout(0.2))
model.add(LSTM(50,return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=100, batch_size=32)

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

plt.plot(data, label='Original Data', color='blue')
plt.plot(np.arange(train_size + time_step, len(data)), predictions, label='Predicted Data', color='red')
plt.legend()
plt.show()
