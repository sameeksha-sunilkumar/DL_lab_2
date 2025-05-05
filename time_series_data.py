import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
data = np.sin(np.arange(0,100,0.1))
data = data.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0,1))
scaler_data = scaler.fit_transform(data)
def create_dataset(data, time_step=1):
    x,y=[],[]
    for i in range(len(data)-time_step-1):
        x.append(data[i:(i+time_step),0])
        y.append(data[i+time_step,0])
    return np.array(x),np.array(y)
time_step=10
x,y = create_dataset(scaler_data,time_step)
x=x.reshape(x.shape[0],x.shape[1],1)
train_size=int(len(x)*0.8)
x_train,x_test=x[:train_size],x[train_size:]
y_train,y_test=y[:train_size],y[train_size:]
model = Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(time_step,1)))
model.add(Dropout(0.2))
model.add(LSTM(50,return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,epochs=100,batch_size=32)
predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)
plt.plot(data,label='og data')
plt.plot(np.arange(train_size+time_step,len(data)-1),predictions,label='predicted data',color='red')
plt.legend()
plt.show()
