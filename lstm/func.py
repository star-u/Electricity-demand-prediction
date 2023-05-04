# coding=gbk
import sys

import pandas as pd
import numpy as np

import pickle
import os

import xlrd
import datetime

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, RepeatVector
from tensorflow.keras.callbacks import History, EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences

city = ["陕西", "铜川", "西安", "渭南", "汉中", "宝鸡", "安康", "咸阳", "西咸", "延安", "榆林"]
type = ".xls"

city_type = []
for _ in range(len(city)):
    city_type.append(city[_] + type)


def date(para):
    delta = pd.Timedelta(str(para) + 'days')
    time = pd.to_datetime('1899-12-30') + delta
    return time


def get_length(city_name):
    file_path = 'data/{}.xls'.format(city_name)
    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_index(0)
    # print(table.row_values(0)[0])
    sum_row = 0
    while table.row_values(sum_row + 1)[0] != 0:
        sum_row += 1

    return sum_row

def get_length2(city_name):
    file_path = '{}.xls'.format(city_name)
    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_index(0)
    # print(table.row_values(0)[0])
    sum_row = 0
    while table.row_values(sum_row + 1)[0] != 0:
        sum_row += 1

    return sum_row

def get_index(city_name, industry_name):
    file_path = 'data/{}.xls'.format(city_name)
    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_index(0)
    for i in range(1, 14):
        print(table.row_values(0)[i])
        if table.row_values(0)[i] == industry_name:
            return i
    return -1

def get_index2(city_name, industry_name):
    file_path = '{}.xls'.format(city_name)
    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_index(0)
    for i in range(1, 14):
        print(table.row_values(0)[i])
        if table.row_values(0)[i] == industry_name:
            return i
    return -1


def get_history(city_name, industry_name):
    file_path = 'data/{}.xls'.format(city_name)
    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_index(0)
    # print(table.row_values(0)[0])
    length = get_length(city_name)
    time_list = []
    for i in range(1, length + 1):
        value = table.row_values(i)[0]
        str_p = str(date(value))
        dateTime_p = datetime.datetime.strptime(str_p, '%Y-%m-%d %H:%M:%S')
        time_list.append(str(dateTime_p)[:10])
    index = get_index(city_name, industry_name)
    electricity = []
    if index == -1:
        print("没有找到该行业，请输入正确的行业进行查询")
    else:
        for i in range(1, length + 1):
            value = table.row_values(i)[index]
            if isinstance(value, str):
                electricity.append(0)
            else:
                electricity.append(value)
    history = []
    for j in range(len(time_list)):
        history.append((time_list[j], electricity[j]))
    return history


def check_file(city_name, industry_name, file_path):
    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_index(0)
    # print(table.row_values(0)[0])
    length = get_length2(city_name)
    
    if length < 30:
        print("长度不对，日期不少于30天")
        return 1
    elif file_path == "":
        print("无输入，请重新输入")
        return 3
    else:
        if file_path not in city_type:
            print("格式不对，请参考输入文件格式输入")
            return 2
        else:
            index = get_index2(city_name, industry_name)
            if index == -1:
                print("没有找到该行业，请输入正确的行业进行查询")
            else:
                for j in range(1, length):
                    if isinstance(table.row_values(j + 1)[index], str):
                        print("文件存在缺失值")
                        return 2

    return 0

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



def get_pred(city_name, industry_name, file_path):
    prediction_tuple = []
    if check_file(city_name, industry_name, file_path) == 0:
        data = xlrd.open_workbook(file_path)
        table = data.sheet_by_index(0)
        length = get_length2(city_name)
        index = get_index2(city_name, industry_name)
        if index == -1:
            print("没有找到该行业，请输入正确的行业进行查询")
        else:
            init_data = np.array(
                ([table.row_values(1)[index]] + [table.row_values(1)[index+ 13]] + table.row_values(1)[
                                                                                            index + 26:])).reshape(
                -1)[np.newaxis, :]

            #ele = np.array([table.row_values(1)[index]])
            for j in range(1, length):
                if not isinstance(table.row_values(j + 1)[index], str) and not isinstance(
                        table.row_values(j + 1)[index + 13], str):
                    init_data = np.concatenate((init_data, np.array(([table.row_values(j + 1)[index]] + [
                        table.row_values(j + 1)[index + 13]] + table.row_values(j + 1)[index + 26:])).reshape(-1)[
                                                           np.newaxis,
                                                           :]),
                                               axis=0)
                    #ele = np.concatenate((ele, np.array([table.row_values(j + 1)[index]])), axis=0)
        MODEL_PATH = 'modelpth2/{}/{}.pth'.format(city_name, industry_name)
        with open(MODEL_PATH, 'rb') as file:
            grid_model = pickle.load(file)
        #testX = np.array([0])
        #print(np.shape(ele[:90]))
        print(init_data.shape)
        init_data = np.expand_dims(init_data,axis=-1)
        init_data = np.expand_dims(init_data,axis=0)
        print(init_data.shape)
        #print(testX.shape)
        max_length = 30
        init_data_padded = pad_sequences(init_data, maxlen=max_length, padding='post', truncating='post')
        #print(init_data_padded)
        prediction = grid_model.predict(init_data_padded)
        print(prediction)
        
        y_pred = np.reshape(prediction, (len(prediction), 1))
        y_pred_list = list(np.squeeze(y_pred))
        print(np.shape(np.array(y_pred_list)))
        
        time_list = []
        for i in range(1, length + 1):
            value = table.row_values(i)[0]
            str_p = str(date(value))
            dateTime_p = datetime.datetime.strptime(str_p, '%Y-%m-%d %H:%M:%S')
            time_list.append(str(dateTime_p)[:10])

        for j in range(max_length):
            prediction_tuple.append((time_list[j], y_pred_list[j]))
    return prediction_tuple


#test = get_history("陕西", "B、城乡居民生活用电合计")
#print(test)
test2 = get_pred("陕西", "B、城乡居民生活用电合计", "陕西.xls")
print(test2)
