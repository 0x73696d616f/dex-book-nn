import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
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
from sklearn.model_selection import train_test_split


def create_lookback(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


# This function takes datasets from the previous function as input and train model using these datasets
def train_model(X_train, Y_train, X_val, Y_val):
    # initialize sequential model, add bidirectional LSTM layer and densely connected output neuron
    model = Sequential()
    model.add(GRU(10, input_shape=(X_train.shape[1], X_train.shape[2]), reset_after=False))
    model.add(Dense(1))
    
    # compile and fit the model
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, Y_train, epochs = 100, batch_size = 16, shuffle = False, 
                    validation_data=(X_val, Y_val), verbose=0,
                    callbacks = [EarlyStopping(monitor='val_loss',min_delta=5e-5,patience=20,verbose=0)])
    return model


# This function uses trained model and validation dataset to calculate RMSE
def get_rmse(model, X_val, Y_val, scaler, look_back = 1):    
    # get predictions and then make some transformations to be able to calculate RMSE properly in USD
    prediction = model.predict(X_val)
    prediction_inverse = scaler.inverse_transform(prediction.reshape(-1, 1))
    Y_val_inverse = scaler.inverse_transform(Y_val.reshape(-1, 1))
    prediction2_inverse = np.array(prediction_inverse[:,0][0:])
    Y_val2_inverse = np.array(Y_val_inverse[:,0])
    
    #calculate RMSE
    RMSE = sqrt(mean_squared_error(Y_val2_inverse, prediction2_inverse))
    return RMSE, prediction2_inverse


dataset_size = 0.3
test_size = 0.1
val_size = 0.2

data = pd.read_csv('bitcoin_price.csv')
data['date'] = pd.to_datetime(data['Timestamp'],unit='s').dt.date
group = data.groupby('date')
Daily_Price = group['Weighted_Price'].mean()
Daily_Price = Daily_Price[-int(dataset_size*len(Daily_Price)):]

# Split the data into training and testing datasets
df_train, df_test = train_test_split(Daily_Price, test_size=test_size, shuffle=False)

# Split the training dataset into training and validation datasets
df_train, df_val = train_test_split(df_train, test_size=val_size, shuffle=False)

working_data = [df_train, df_val, df_test]
working_data = pd.concat(working_data)

working_data = working_data.reset_index()
working_data['date'] = pd.to_datetime(working_data['date'])
working_data = working_data.set_index('date')

training_set = df_train.values
training_set = np.reshape(training_set, (len(training_set), 1))
validation_set = df_val.values
validation_set = np.reshape(validation_set, (len(validation_set), 1))
test_set = df_test.values
test_set = np.reshape(test_set, (len(test_set), 1))

#scale datasets
scaler = MinMaxScaler()
training_set = scaler.fit_transform(training_set)
validation_set = scaler.fit_transform(validation_set)
test_set = scaler.fit_transform(test_set)

# create datasets which are suitable for time series forecasting
look_back = 10
X_train, Y_train = create_lookback(training_set, look_back)
X_val, Y_val = create_lookback(validation_set, look_back)
X_test, Y_test = create_lookback(test_set, look_back)


# reshape datasets so that they will be ok for the requirements of the LSTM model in Keras
X_train = np.reshape(X_train, (len(X_train), X_train.shape[1], 1))
X_val = np.reshape(X_val, (len(X_val), X_val.shape[1], 1))
X_test = np.reshape(X_test, (len(X_test), X_test.shape[1], 1))

# get predictions and then make some transformations to be able to calculate RMSE properly in USD
Y_test_inverse = scaler.inverse_transform(Y_test.reshape(-1, 1))
Y_test2_inverse = np.array(Y_test_inverse[:,0])
Test_Dates = Daily_Price[int((1-test_size)*len(Daily_Price)):].index

model = train_model(X_train, Y_train, X_val, Y_val)
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
py.plot(fig, filename='results')

# Convert the model architecture to JSON
model_json = model.to_json()

# Get the model weights and convert them to a list
weights_list = [weight.tolist() for weight in model.get_weights()]

# Create a dictionary to store the model architecture and weights
model_data = {"model": model_json, "weights": weights_list}

# Define the maximum file size
max_file_size = 11000

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
