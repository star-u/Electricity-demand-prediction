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
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import warnings
import json

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

city = ["陕西", "铜川", "西安", "渭南", "汉中", "宝鸡", "安康", "咸阳", "西咸", "延安", "榆林","商洛"]
type = ".xls"
industories = ['B、城乡居民生活用电合计', '一、农、林、牧、渔业', '七、住宿和餐饮业', '三、建筑业', '九、房地产业', '二、工业',
               '五、信息传输、软件和信息技术服务业', '全社会用电', '八、金融业', '六、批发和零售业',
               '十、租赁和商务服务业', '十一、公共服务及管理组织', '四、交通运输、仓储和邮政业']
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


def get_length2(city_name, file_path):
    #file_path = '{}.xls'.format(city_name)
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


def get_index2(city_name, industry_name,file_path):
    #file_path = '{}.xls'.format(city_name)
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


def check_file(city_name, file_path):
    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_index(0)
    # print(table.row_values(0)[0])
    length = get_length2(city_name, file_path)

    if length < 30:
        print("日期小于30天")
        return 1
    elif file_path == "":
        print("文件不为空")
        return 2

    return 0


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


@app.route('/pred', methods=['GET', 'POST'])
def get_pred():
    '''
    if request.method == 'POST':
        elec_file = request.files['elec']
        city_name = request.form['city_name']
        print(request.files)
        if (len(elec_file.filename) > 0):
            file_path = city_name + elec_file.filename
            elec_file.save(file_path)
        else:
            return jsonify('file upload error')
    else:
        return render_template('test_6.html')
    '''
    file_path = "安康安康.xls"
    city_name = "安康"
    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_index(0)
    length = get_length2(city_name, file_path)

    res = []
    for industry_name in industories:
        prediction_tuple = []
        if check_file(city_name, file_path) == 0:

            index = get_index2(city_name, industry_name, file_path)
            init_data = np.array(
                ([table.row_values(1)[index]] + [table.row_values(1)[index + 13]] + table.row_values(1)[
                                                                                    27:])).reshape(
                -1)[np.newaxis, :]

            # ele = np.array([table.row_values(1)[index]])
            for j in range(1, length):
                if isinstance(table.row_values(j + 1)[index], str):
                    table.row_values(j + 1)[index] = 0
                init_data = np.concatenate((init_data, np.array(([table.row_values(j + 1)[index]] + [
                    table.row_values(j + 1)[index + 13]] + table.row_values(j + 1)[27:])).reshape(-1)[
                                                       np.newaxis,
                                                       :]),
                                           axis=0)

                    # ele = np.concatenate((ele, np.array([table.row_values(j + 1)[index]])), axis=0)
            MODEL_PATH = 'modelpth2/{}/{}.pth'.format(city_name, industry_name)
            with open(MODEL_PATH, 'rb') as file:
                grid_model = pickle.load(file)
            # testX = np.array([0])
            # print(np.shape(ele[:90]))
            #print(init_data.shape)
            init_data = np.expand_dims(init_data, axis=-1)
            init_data = np.expand_dims(init_data, axis=0)
            #print(init_data.shape)
            # print(testX.shape)
            max_length = 30
            init_data_padded = pad_sequences(init_data, maxlen=max_length, padding='post', truncating='post')
            prediction = grid_model.predict(init_data_padded)
            #print(prediction.shape)

            y_pred = np.reshape(prediction, (len(prediction), 1))
            y_pred_list = list(np.squeeze(y_pred))
            print(np.shape(np.array(y_pred_list)))

            time_list = []
            for i in range(1, length + 1):
                value = table.row_values(i)[0]
                str_p = str(date(value))
                dateTime_p = datetime.datetime.strptime(str_p, '%Y-%m-%d %H:%M:%S')
                time_list.append(str(dateTime_p)[:10])

            for j in range(len(y_pred_list)):
                prediction_tuple.append((time_list[length-30+j], str(y_pred_list[j])))
            print(prediction_tuple)
            res.append(prediction_tuple)
            
        else:
            return json.dumps(check_file(city_name, file_path))
    return json.dumps(res)

if __name__ == '__main__':
    m = get_pred()
    print(m)
