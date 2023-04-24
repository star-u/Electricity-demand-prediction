# coding=gbk
import xlrd

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, LSTM
#import torch
import pickle
import os

city = ["陕西", "铜川", "西安", "渭南", "汉中", "宝鸡", "安康", "咸阳",  "西咸", "延安", "榆林"]
length = 1124
def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i,0])
    return np.array(dataX),np.array(dataY)

def build_model(optimizer, city_num):
    file_path = 'data/{}.xls'.format(city[city_num])

    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_index(0)
    

    industry_name = table.row_values(0)[i+1] 

    init_data = np.array(([table.row_values(1)[i+1]] + [table.row_values(1)[i+1+13]] + table.row_values(1)[i+1+26:] )).reshape(-1)[np.newaxis, :]
    if city_num>7:
        length = 969
    for j in range(1,length):
        init_data = np.concatenate((init_data,np.array(([table.row_values(j+1)[i+1]] + [table.row_values(j+1)[i+1+13]] + table.row_values(j+1)[i+1+26:] )).reshape(-1)[np.newaxis, :]),axis=0)

    k = 0.8  
    train_size = int(len(init_data) * k)
    feature_num = init_data.shape[1] 
    train_step = 30 
    
    grid_model = Sequential()
    grid_model.add(LSTM(128,return_sequences=True,input_shape=(train_step, feature_num))) 
    grid_model.add(LSTM(128))
    grid_model.add(Dropout(0.2))
    grid_model.add(Dense(1))

    grid_model.compile(loss = 'mse',optimizer = optimizer)
    return grid_model

def LSTM_model(i, city_num):

    file_path = 'data/{}.xls'.format(city[city_num])

    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_index(0)


    industry_name = table.row_values(0)[i+1] 

    init_data = np.array(([table.row_values(1)[i+1]] + [table.row_values(1)[i+1+13]] + table.row_values(1)[i+1+26:] )).reshape(-1)[np.newaxis, :]
    if city_num>7:
        length = 969
    for j in range(1,length): 
        init_data = np.concatenate((init_data,np.array(([table.row_values(j+1)[i+1]] + [table.row_values(j+1)[i+1+13]] + table.row_values(j+1)[i+1+26:] )).reshape(-1)[np.newaxis, :]),axis=0)

    k = 0.8  
    train_size = int(len(init_data) * k)
    feature_num = init_data.shape[1] 
    train_step = 30 

    df_for_training = init_data[:train_size] 
    df_for_testing = init_data[train_size:]
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    df_for_training_scaled = scaler.fit_transform(df_for_training) 

    df_for_testing_scaled = scaler.transform(df_for_testing)

    trainX, trainY = createXY(df_for_training_scaled, train_step)

    testX,testY=createXY(df_for_testing_scaled,train_step)


    grid_model = KerasRegressor(build_fn=build_model,verbose=2,validation_data=(testX,testY),batch_size=64,epochs=200,optimizer='adam',city_num=city_num)
    grid_model.fit(trainX, trainY)
    folder_path = "modelpth/{}".format(city[city_num])
    os.makedirs(folder_path, exist_ok=True)
    MODEL_PATH = 'modelpth/{}/{}.pth'.format(city[city_num], industry_name)
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(grid_model, file)
    #torch.save(grid_model, MODEL_PATH)
    
    #return grid_model

def test_model(i, city_num):
    file_path = 'data/{}.xls'.format(city[city_num])

    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_index(0)

    industry_name = table.row_values(0)[i+1] 
    init_data = np.array(([table.row_values(1)[i+1]] + [table.row_values(1)[i+1+13]] + table.row_values(1)[i+1+26:] )).reshape(-1)[np.newaxis, :]
    if city_num>7:
        length = 969
    for j in range(1,length): 
        init_data = np.concatenate((init_data,np.array(([table.row_values(j+1)[i+1]] + [table.row_values(j+1)[i+1+13]] + table.row_values(j+1)[i+1+26:] )).reshape(-1)[np.newaxis, :]),axis=0)

    k = 0.8  
    train_size = int(len(init_data) * k)
    feature_num = init_data.shape[1] 
    train_step = 30 

    df_for_training = init_data[:train_size] 
    df_for_testing = init_data[train_size:]
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    df_for_training_scaled = scaler.fit_transform(df_for_training) 

    df_for_testing_scaled = scaler.transform(df_for_testing)

    trainX, trainY = createXY(df_for_training_scaled, train_step)

    testX,testY=createXY(df_for_testing_scaled,train_step)
    MODEL_PATH = 'modelpth/{}/{}.pth'.format(city[city_num], industry_name)
    with open(MODEL_PATH, 'rb') as file:
        grid_model=pickle.load(file)
    #grid_model = torch.load('modelpth\{}.pth'.format(industry_name))
    
    prediction = grid_model.predict(testX)
    prediction_copies_array = np.repeat(prediction, feature_num, axis=-1)


    y_pred = scaler.inverse_transform(np.reshape(prediction_copies_array, (len(prediction), feature_num)))[:, 0]
    original_copies_array = np.repeat(testY, feature_num, axis=-1)
    y_act=scaler.inverse_transform(np.reshape(original_copies_array,(len(testY), feature_num)))[:,0]

    error1 = 0  
    error2 = 0  
 
    y_pred_sum = 0
    y_act_sum = 0
    for j in range(13,(13+31)): 
        y_pred_sum += y_pred[j]
        y_act_sum += y_act[j]
        error2 += abs((y_pred[j] - y_act[j]) / y_act[j])
    error1 = (y_pred_sum - y_act_sum) / y_act_sum
    error2 = error2 / 31
    print(f'error1 in {industry_name} and 8 is {error1}')
    print(f'error2 in {industry_name} and 8 is {error2}')

    print(f'prediction for {industry_name} in 8 is : {list(y_pred[13:(13+31)])}')


for city_num in range(8, len(city)):
    for i in range(13):
        LSTM_model(i, city_num)
        test_model(i, city_num)
