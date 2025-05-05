import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('AirPassengers.csv', usecols=[1])
dataset = data.values

sc = MinMaxScaler(feature_range=(0, 1))
scaled_data = sc.fit_transform(dataset)

x_train = []
y_train = []
look_back = 5
for i in range(look_back, len(scaled_data)):
    x_train.append(scaled_data[i-look_back:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(units=50))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=50, batch_size=32)

predicted = model.predict(x_train)
predicted = sc.inverse_transform(predicted)

plt.plot(dataset, label='Original')
plt.plot(predicted, label='Predicted', color='red')
plt.legend()
plt.title('Air Passengers Prediction')
plt.show()
