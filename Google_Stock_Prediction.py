import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout 
import matplotlib.pyplot as plt

dataset= pd.read_csv("D:\My Stuff\Arjun\FDP\Google_Stock_Price_Train.csv")
training_data = dataset.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
mScaler = MinMaxScaler()
training_data= mScaler.fit_transform(training_data)
x_train = training_data[:1257] 
y_train = training_data[1:]

x_train = np.reshape(x_train, (1257,1,1))

rnnRegressor = Sequential()

rnnRegressor.add(LSTM(units = 4, activation='sigmoid',input_shape=(None,1)))

rnnRegressor.add(Dense(units=1))

rnnRegressor.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])

rnnRegressor.fit(x_train, y_train,epochs=1000, batch_size=32)




data_test = pd.read_csv("D:\My Stuff\Arjun\FDP\Google_Stock_Price_Test.csv")
test_data = data_test.iloc[:,1:2].values
test_data= mScaler.transform(test_data)

newData = np.reshape(test_data,(20,1,1))
prediction= rnnRegressor.predict(newData)

OrgPredict = mScaler.inverse_transform(prediction)

plt.plot(mScaler.inverse_transform(test_data), color='green', label='Prediction')
plt.plot(OrgPredict, color='red', label='Prediction')
plt.legend()
















