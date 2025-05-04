import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
df=pd.read_csv('AirPassengers.csv', usecols=[1])
scaler=MinMaxScaler(feature_range=(0,1))
df=scaler.fit_transform(df)
plt.plot(df)
plt.show()
train_size=int(len(df)*0.7)
train,test=df[0:train_size],df[train_size:]
def create_dataset(dataset,look_back=1):
    x,y=[],[]
    for i in range(len(dataset)-look_back-1):
        x.append(dataset[i:(i+look_back),0])
        y.append(dataset[i+look_back,0])
    return np.array(x),np.array(y)
look_back=5
train_x,train_y=create_dataset(train,look_back)
test_x,test_y=create_dataset(test,look_back)
train_x=np.reshape(train_x,(train_x.shape[0],1,train_x.shape[1]))
test_x=np.reshape(test_x,(test_x.shape[0],1,test_x.shape[1]))
model=Sequential()
model.add(LSTM(8,input_shape=(1,look_back)))
model.add(Dense(8))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
history=model.fit(train_x,train_y,epochs=100,batch_size=1)
trainPredict=model.predict(train_x)
testPredict=model.predict(test_x)
trainScore=np.sqrt(mean_squared_error(train_y,trainPredict[:,0]))
testScore = np.sqrt(mean_squared_error(test_y, testPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
trainPredictPlot = np.empty_like(df)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, 0] = trainPredict[:,0]
testPredictPlot = np.empty_like(df)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(df)-1, 0] = testPredict[:,0]
plt.plot(scaler.inverse_transform(df))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.title('air passenger prediction')
plt.xlabel('time')
plt.ylabel('num of passengers')
plt.show()
