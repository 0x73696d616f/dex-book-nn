import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error
from math import sqrt
from random import randint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from keras.callbacks import EarlyStopping
import plotly.offline as py
import plotly.graph_objs as go
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
import json
import os
from datetime import date


def create_lookback(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


# This function takes datasets from the previous function as input and train model using these datasets
def train_model(X_train, Y_train, X_test, Y_test):
    # initialize sequential model, add bidirectional LSTM layer and densely connected output neuron
    model = Sequential()
    model.add(GRU(128, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    
    # compile and fit the model
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, Y_train, epochs = 100, batch_size = 30, shuffle = False, 
                    validation_data=(X_test, Y_test), verbose=0,
                    callbacks = [EarlyStopping(monitor='val_loss',min_delta=5e-5,patience=20,verbose=0)])
    return model


# This function uses trained model and test dataset to calculate RMSE
def get_rmse(model, X_test, Y_test, scaler, look_back = 1):    
    # get predictions and then make some transformations to be able to calculate RMSE properly in USD
    prediction = model.predict(X_test)
    prediction_inverse = scaler.inverse_transform(prediction.reshape(-1, 1))
    Y_test_inverse = scaler.inverse_transform(Y_test.reshape(-1, 1))
    prediction2_inverse = np.array(prediction_inverse[:,0][0:])
    Y_test2_inverse = np.array(Y_test_inverse[:,0])
    
    #calculate RMSE
    RMSE = sqrt(mean_squared_error(Y_test2_inverse, prediction2_inverse))
    return RMSE, prediction2_inverse


data = pd.read_csv('bitcoin_price.csv')
data['date'] = pd.to_datetime(data['Timestamp'],unit='s').dt.date
group = data.groupby('date')
Daily_Price = group['Weighted_Price'].mean()

print(Daily_Price)

d0 = date(2016, 1, 1)
d1 = date(2017, 10, 15)
delta = d1 - d0
days_look = delta.days + 1
print("days_look: ", days_look)

d0 = date(2017, 8, 21)
d1 = date(2017, 10, 20)
delta = d1 - d0
days_from_train = delta.days + 1
print("days_from_train: ", days_from_train)

d0 = date(2017, 10, 15)
d1 = date(2017, 10, 20)
delta = d1 - d0
days_from_end = delta.days + 1
print("days_from_end: ", days_from_end)

df_train= Daily_Price[len(Daily_Price)-days_look-days_from_end:len(Daily_Price)-days_from_train]
df_test= Daily_Price[len(Daily_Price)-days_from_train:]

working_data = [df_train, df_test]
working_data = pd.concat(working_data)

working_data = working_data.reset_index()
working_data['date'] = pd.to_datetime(working_data['date'])
working_data = working_data.set_index('date')

df_train = working_data[:-60]
df_test = working_data[-60:]

training_set = df_train.values
training_set = np.reshape(training_set, (len(training_set), 1))
test_set = df_test.values
test_set = np.reshape(test_set, (len(test_set), 1))

#scale datasets
scaler = MinMaxScaler()
training_set = scaler.fit_transform(training_set)
test_set = scaler.fit_transform(test_set)

# create datasets which are suitable for time series forecasting
look_back = 1
X_train, Y_train = create_lookback(training_set, look_back)
X_test, Y_test = create_lookback(test_set, look_back)

# reshape datasets so that they will be ok for the requirements of the LSTM model in Keras
X_train = np.reshape(X_train, (len(X_train), X_train.shape[1], 1))
X_test = np.reshape(X_test, (len(X_test), X_test.shape[1], 1))

# get predictions and then make some transformations to be able to calculate RMSE properly in USD
Y_test_inverse = scaler.inverse_transform(Y_test.reshape(-1, 1))
Y_test2_inverse = np.array(Y_test_inverse[:,0])
Test_Dates = Daily_Price[len(Daily_Price)-days_from_train:].index

model = train_model(X_train, Y_train, X_test, Y_test)
RMSE, predictions = get_rmse(model, X_test, Y_test, scaler, look_back)
print('Test GRU model RMSE: %.3f' % RMSE)

predictions_new = predictions

trace1 = go.Scatter(x=Test_Dates[:-1], y=Y_test2_inverse[:-1], name= 'Actual Price', 
                   line = dict(color = ('rgb(66, 244, 155)'),width = 2))
trace2 = go.Scatter(x=Test_Dates[:-1], y=predictions_new[1:], name= 'Predicted Price',
                   line = dict(color = ('rgb(244, 146, 65)'),width = 2))
data = [trace1, trace2]
layout = dict(title = 'Comparison of true prices (on the test dataset) with prices our model predicted, by dates',
             xaxis = dict(title = 'Date'), yaxis = dict(title = 'Price, USD'))
fig = dict(data=data, layout=layout)
py.plot(fig, filename='results_demonstrating2')

# Convert the model architecture to JSON
model_json = model.to_json()

# Get the model weights and convert them to a list
weights_list = [weight.tolist() for weight in model.get_weights()]

# Create a dictionary to store the model architecture and weights
model_data = {"model": model_json, "weights": weights_list}

# Define the maximum file size in bytes (130 KB)
max_file_size = 10000

# Calculate the number of chunks required
total_chunks = int(np.ceil(len(json.dumps(model_data).encode('utf-8')) / max_file_size))

# Create a directory to store the model data chunks
os.makedirs('model_chunks', exist_ok=True)

# Split the model data into chunks
for i in range(total_chunks):
    # Calculate the start and end indices for each chunk
    start_index = i * max_file_size
    end_index = (i + 1) * max_file_size

    # Get the chunk data
    chunk_data = json.dumps(model_data).encode('utf-8')[start_index:end_index]

    # Write the chunk data to a file
    with open(f'model_chunks/model_data_{i}.json', 'wb') as file:
        file.write(chunk_data)

# Load the model architecture and weights from the file chunks
loaded_model_data = bytearray()
for i in range(total_chunks):
    # Read the chunk data from the file
    with open(f'model_chunks/model_data_{i}.json', 'rb') as file:
        chunk_data = file.read()
        loaded_model_data.extend(chunk_data)

# Append the chunk data to the loaded model data dictionary
loaded_model_data = json.loads(loaded_model_data.decode('utf-8'))

# Create a new model from the loaded architecture
model_loaded = model_from_json(loaded_model_data['model'])

# Set the loaded weights to the model
loaded_weights = [np.array(weight) for weight in loaded_model_data['weights']]
model_loaded.set_weights(loaded_weights)
