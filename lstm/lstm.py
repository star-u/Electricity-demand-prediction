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

city = ["陕西", "铜川", "西安", "渭南", "汉中", "宝鸡", "安康", "咸阳", "西咸", "延安", "榆林"]


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
    time_step = 90
    for i in range(len(dataset)-n_past-time_step):
        dataX.append(dataset[i:i+n_past, :])
        dataY.append(dataset[i+n_past:i+n_past+time_step, 0])
    return np.array(dataX), np.array(dataY)


def build_model():

    grid_model = Sequential()
    #CNN
    grid_model.add(Conv2D(
        filters=32,
        kernel_size=(3,3),
        activation=('relu'),
        padding='SAME',
        input_shape=(180, 13, 1),
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)))

    grid_model.add(MaxPooling2D(pool_size=(2,2)))

    grid_model.add(Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation=('relu'),
        padding='SAME'))

    grid_model.add(Flatten())
    
    grid_model.add(RepeatVector(90))
    grid_model.add(LSTM(128, return_sequences=False))
    grid_model.add(Dropout(0.2))

    grid_model.add(Dense(90))
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
    grid_model.summary()
    return grid_model


def LSTM_model(i, city_num):
    file_path = 'data/{}.xls'.format(city[city_num])
    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_index(0)
    industry_name = table.row_values(0)[i + 1]

    init_data = np.array(
        ([table.row_values(1)[i + 1]] + [table.row_values(1)[i + 1 + 13]] + table.row_values(1)[i + 1 + 26:])).reshape(
        -1)[np.newaxis, :]
    length = get_length(city_num)
    for j in range(1, length):
        init_data = np.concatenate((init_data, np.array(([table.row_values(j + 1)[i + 1]] + [
            table.row_values(j + 1)[i + 1 + 13]] + table.row_values(j + 1)[i + 1 + 26:])).reshape(-1)[np.newaxis, :]),
                                   axis=0)

    k = 1
    train_size = int(len(init_data) * k)
    feature_num = init_data.shape[1]
    train_step = 180

    df_for_training = init_data[:train_size]
    df_for_testing = init_data[train_size:]
    trainX, trainY = createXY(df_for_training, train_step)
    testX, testY = createXY(df_for_testing, train_step)
    print(init_data.shape, trainX.shape, trainY.shape)
    #sys.exit()
    max_length = 180
    trainX = pad_sequences(trainX, maxlen=max_length, padding='post', truncating='post')
    grid_model = KerasRegressor(build_fn=build_model, verbose=2, validation_data=(testX, testY), batch_size=64,
                                epochs=200)
    grid_model.fit(trainX, trainY)

    folder_path = "modelpth/{}".format(city[city_num])
    os.makedirs(folder_path, exist_ok=True)
    MODEL_PATH = 'modelpth/{}/{}.pth'.format(city[city_num], industry_name)
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(grid_model, file)
    # torch.save(grid_model, MODEL_PATH)

    # return grid_model


def test_model(i, city_num):
    file_path = 'data/{}.xls'.format(city[city_num])

    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_index(0)
    industry_name = table.row_values(0)[i + 1]
    init_data = np.array(
        ([table.row_values(1)[i + 1]] + [table.row_values(1)[i + 1 + 13]] + table.row_values(1)[i + 1 + 26:])).reshape(
        -1)[np.newaxis, :]
    length = get_length(city_num)
    for j in range(1, length):
        init_data = np.concatenate((init_data, np.array(([table.row_values(j + 1)[i + 1]] + [
            table.row_values(j + 1)[i + 1 + 13]] + table.row_values(j + 1)[i + 1 + 26:])).reshape(-1)[np.newaxis, :]),
                                   axis=0)

    k = 0.8
    train_size = int(len(init_data) * k)
    feature_num = init_data.shape[1]
    train_step = 30
    df_for_testing = init_data[train_size:]

    testX, testY = createXY(df_for_testing, train_step)
    MODEL_PATH = 'modelpth/{}/{}.pth'.format(city[city_num], industry_name)
    with open(MODEL_PATH, 'rb') as file:
        grid_model = pickle.load(file)
    # grid_model = torch.load('modelpth\{}.pth'.format(industry_name))
    print(testX.shape)
    prediction = grid_model.predict(testX)
    print(prediction.shape)
    y_pred = np.reshape(prediction, (len(prediction), 90))
    y_act = np.reshape(testY, (len(testY), 90))

    error1 = 0
    error2 = 0

    y_pred_sum = 0
    y_act_sum = 0
    for j in range(13, (13 + 31)):
        y_pred_sum += y_pred[j]
        y_act_sum += y_act[j]
        error2 += abs((y_pred[j] - y_act[j]) / y_act[j])
    error1 = (y_pred_sum - y_act_sum) / y_act_sum
    error2 = error2 / 31
    print(f'error1 in {industry_name} and 8 is {error1}')
    print(f'error2 in {industry_name} and 8 is {error2}')

   # print(f'prediction for {industry_name} in 8 is : {list(y_pred[13:(13 + 31)])}')


for city_num in range(len(city)):
    for i in range(13):
        LSTM_model(i, city_num)
        #test_model(i, city_num)
        sys.exit()
