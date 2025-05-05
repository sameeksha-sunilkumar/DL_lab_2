import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

data = np.sin(np.arange(0, 100, 0.1)).reshape(-1, 1)

sc = MinMaxScaler(feature_range=(0, 1))
scaled_data = sc.fit_transform(data)

x_train = []
y_train = []
time_step = 10
for i in range(time_step, len(scaled_data)):
    x_train.append(scaled_data[i-time_step:i, 0])
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

plt.plot(data, label='Original')
plt.plot(predicted, label='Predicted', color='red')
plt.legend()
plt.title('Sine Wave Prediction')
plt.show()
