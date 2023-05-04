# coding=gbk
import sys

import xlrd

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
# import torch
import pickle
import os
from keras.layers import Masking, Embedding
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, RepeatVector
from tensorflow.keras.callbacks import History, EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences

import warnings

warnings.filterwarnings('ignore')

city = ["陕西", "铜川", "西安", "渭南", "汉中", "宝鸡", "安康", "咸阳", "西咸", "延安", "榆林","商洛"]


def get_length(city_num):
    file_path = 'data/{}.xls'.format(city[city_num])
    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_index(0)
    # print(table.row_values(0)[0])
    sum_row = 0
    while table.row_values(sum_row + 1)[0] != 0:
        sum_row += 1

    return sum_row


def createXY(dataset, n_past):
    dataX = []
    dataY = []
    time_step = 30
    for i in range(len(dataset)-n_past-time_step):
        dataX.append(dataset[i:i+n_past,:])
        dataY.append(dataset[i+n_past:i+n_past+time_step, 0])
    return np.array(dataX), np.array(dataY)


def build_model():

    grid_model = Sequential()
    #CNN
    time_step = 30
    grid_model.add(LSTM(128, input_shape=(time_step,13),return_sequences=True))
    grid_model.add(LSTM(128))
    grid_model.add(Dropout(0.2))

    grid_model.add(Dense(time_step))
    grid_model.add(Activation("relu"))
    '''
    #LSTM
    grid_model.add(Masking(mask_value= -1,input_shape=(sequenceLength, 13)))
    grid_model.add(LSTM(128,  input_shape=(sequenceLength, 13)))

    #grid_model.add(LSTM(128, return_sequences=True))
    #grid_model.add(LSTM(128))
    grid_model.add(Dropout(0.2))
    grid_model.add(Dense(90))
    '''
    grid_model.compile(loss='mse', optimizer='adam')
    
    return grid_model


def LSTM_model(i, city_num):
    file_path = 'data/{}.xls'.format(city[city_num])
    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_index(0)
    industry_name = table.row_values(0)[i + 1]

    init_data = np.array(
        ([table.row_values(1)[i + 1]] + [table.row_values(1)[i + 1 + 13]] + table.row_values(1)[27:27+11])).reshape(
        -1)[np.newaxis, :]

    length = get_length(city_num)
    for j in range(1, length):
        if  not isinstance(table.row_values(j+1)[i+1],str) and not isinstance(table.row_values(j+1)[i+1+13],str):
            init_data = np.concatenate((init_data, np.array(([table.row_values(j + 1)[i + 1]] + [table.row_values(j + 1)[i + 1 + 13]] + table.row_values(j + 1)[27:27+11])).reshape(-1)[np.newaxis, :]),
                                   axis=0)

    k = 1
    train_size = int(len(init_data) * k)
    train_step = 30

    df_for_training = init_data[:train_size]
    df_for_testing = init_data[train_size:]
    trainX, trainY = createXY(df_for_training, train_step)
    testX, testY = createXY(df_for_testing, train_step)
    print(init_data.shape, trainX.shape, trainY.shape)
    #sys.exit()
    #max_length = 180
    #trainX = pad_sequences(trainX, maxlen=max_length, padding='post', truncating='post')
    grid_model = KerasRegressor(build_fn=build_model, verbose=2, validation_data=(testX, testY), batch_size=64,
                                epochs=20)
    grid_model.fit(trainX, trainY)

    folder_path = "modelpth2/{}".format(city[city_num])
    os.makedirs(folder_path, exist_ok=True)
    MODEL_PATH = 'modelpth2/{}/{}.pth'.format(city[city_num], industry_name)
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(grid_model, file)
    # torch.save(grid_model, MODEL_PATH)

    # return grid_model



for city_num in range(11,len(city)):
    for i in range(13):
        LSTM_model(i, city_num)
        #test_model(i, city_num)
        #sys.exit()
