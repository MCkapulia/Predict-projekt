from datetime import date
from datetime import timedelta
import pandas as pd
import yfinance as yf
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from price import tinkoffPrice
from keras.models import load_model

START = "2006-05-01"
TODAY = date.today().strftime("%Y-%m-%d")

def load_Data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

def train_test_Data(ticker):
    data = load_Data(ticker)
    train = pd.DataFrame(data[0:int(len(data) * 0.70)])
    test = pd.DataFrame(data[int(len(data) * 0.70): int(len(data))])
    train.head()
    test.head()
    train_close = train.iloc[:, 4:5].values
    test_close = test.iloc[:, 4:5].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(train_close)
    return data_training_array

def normalize_Data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    norm_data = scaler.fit_transform(data)
    return norm_data

def conversion_Data(data):
    x_train = []
    y_train = []

    for i in range(100, data.shape[0]):
        x_train.append(data[i - 100: i])
        y_train.append(data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    return x_train, y_train

def create_model(ticker):
    data = train_test_Data(ticker)
    x_train, y_train = conversion_Data(data)
    model = Sequential()
    model.add(LSTM(units=100, activation='tanh', return_sequences=True
                   , input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.1))

    model.add(LSTM(units=200, activation='tanh'))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))
    model.summary()
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[tf.keras.metrics.MeanAbsoluteError()])
    model.fit(x_train, y_train, epochs=20)
    model.save('my_model.keras')

def predict_1day():
    scaler = MinMaxScaler(feature_range=(0, 1))
    tinkoffPrice.run()
    xl = pd.read_excel("data.xlsx")
    xl = xl[['close']]
    window_size = 100
    test_df = xl.values
    data_our_array = scaler.fit_transform(test_df)
    data_our_array = pd.DataFrame(data_our_array)
    recent_data = np.resize(np.array(data_our_array.tail(window_size + 1)), window_size).reshape(1, window_size, 1)
    model = keras.models.load_model('D:\\Проги\\pythonProject\\Predict\\price\\my_model.keras')
    predicted_value_scaled = model.predict(recent_data)
    predicted_value = scaler.inverse_transform(predicted_value_scaled)
    return predicted_value[0][0]

def predict_2day():
    predict_1 = [predict_1day()]
    scaler = MinMaxScaler(feature_range=(0, 1))
    tinkoffPrice.run()
    xl = pd.read_excel("data.xlsx")
    xl = xl[['close']]
    window_size = 100
    test_df = xl.values
    test_df = delete_element(test_df, predict_1[0])
    data_our_array = scaler.fit_transform(test_df)
    data_our_array = pd.DataFrame(data_our_array)
    recent_data = np.resize(np.array(data_our_array.tail(window_size + 1)), window_size).reshape(1, window_size, 1)
    model = keras.models.load_model('D:\\Проги\\pythonProject\\Predict\\price\\my_model.keras')
    predicted_value_scaled = model.predict(recent_data)
    predicted_value = scaler.inverse_transform(predicted_value_scaled)
    predict_1.append(predicted_value)
    return predict_1

def delete_element(test_df, predict):
    test_df = test_df.tolist()
    del test_df[0]
    predict_1=[]
    predict_1.append(float(predict))
    test_df.append(predict_1)
    test_df = np.array(test_df)
    return test_df

def predict_3day():
    predict_2 = predict_2day()
    scaler = MinMaxScaler(feature_range=(0, 1))
    tinkoffPrice.run()
    xl = pd.read_excel("data.xlsx")
    xl = xl[['close']]
    window_size = 100
    test_df = xl.values
    test_df = delete_element(test_df, predict_2[0])
    test_df = delete_element(test_df, predict_2[1])
    data_our_array = scaler.fit_transform(test_df)
    data_our_array = pd.DataFrame(data_our_array)
    recent_data = np.resize(np.array(data_our_array.tail(window_size + 1)), window_size).reshape(1, window_size, 1)
    model = keras.models.load_model('D:\\Проги\\pythonProject\\Predict\\price\\my_model.keras')
    predicted_value_scaled = model.predict(recent_data)
    predicted_value = scaler.inverse_transform(predicted_value_scaled)
    predict_2.append(predicted_value)
    return predict_2

def predict_4day():
    predict_3 = predict_3day()
    scaler = MinMaxScaler(feature_range=(0, 1))
    tinkoffPrice.run()
    xl = pd.read_excel("data.xlsx")
    xl = xl[['close']]
    window_size = 100
    test_df = xl.values
    test_df = delete_element(test_df, predict_3[0])
    test_df = delete_element(test_df, predict_3[1])
    test_df = delete_element(test_df, predict_3[2])
    data_our_array = scaler.fit_transform(test_df)
    data_our_array = pd.DataFrame(data_our_array)
    recent_data = np.resize(np.array(data_our_array.tail(window_size + 1)), window_size).reshape(1, window_size, 1)
    model = keras.models.load_model('D:\\Проги\\pythonProject\\Predict\\price\\my_model.keras')
    predicted_value_scaled = model.predict(recent_data)
    predicted_value = scaler.inverse_transform(predicted_value_scaled)
    predict_3.append(predicted_value)
    return predict_3
def predict_5day():
    predict_4 = predict_4day()
    scaler = MinMaxScaler(feature_range=(0, 1))
    tinkoffPrice.run()
    xl = pd.read_excel("data.xlsx")
    xl = xl[['close']]
    window_size = 100
    test_df = xl.values
    test_df = delete_element(test_df, predict_4[0])
    test_df = delete_element(test_df, predict_4[1])
    test_df = delete_element(test_df, predict_4[2])
    test_df = delete_element(test_df, predict_4[3])
    data_our_array = scaler.fit_transform(test_df)
    data_our_array = pd.DataFrame(data_our_array)
    recent_data = np.resize(np.array(data_our_array.tail(window_size + 1)), window_size).reshape(1, window_size, 1)
    model = keras.models.load_model('D:\\Проги\\pythonProject\\Predict\\price\\my_model.keras')
    predicted_value_scaled = model.predict(recent_data)
    predicted_value = scaler.inverse_transform(predicted_value_scaled)
    predict_4.append(predicted_value)
    return predict_4

