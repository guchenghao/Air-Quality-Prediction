#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /Users/guchenghao/Desktop/5002Projcet/main.py
# Project: /Users/guchenghao/Desktop/5002Projcet
# Created Date: Wednesday, December 5th 2018, 2:05:38 pm
# Author: Harold Gu
# -----
# Last Modified: Wednesday, 5th December 2018 2:05:38 pm
# Modified By: Harold Gu
# -----
# Copyright (c) 2018 HKUST
# #
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
###
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as son
from scipy.stats import skew
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import math
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import copy
from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute, IterativeImputer


# ! 评估指标函数
def smape(actual, predicted):
    dividend = np.abs(np.array(actual) - np.array(predicted))
    c = np.array(actual) + np.array(predicted)

    return (2 / len(predicted)) * np.mean(np.divide(dividend, c, out=np.zeros_like(dividend), where=c != 0, casting='unsafe'))


def smape_new(x, y):
    x = x.tolist()
    y = y.tolist()
    if len(x) != len(y):
        return('error of length')
    else:
        n = len(x)
        diff = []
        for i in range(n):
            diff.append(abs(x[i] - y[i]))
        value = 2 * sum(diff) / (sum(x) + sum(y))
        return value/n


predict_final = pd.read_csv('drive/prediction_final.csv')
final_train = pd.read_csv('drive/final.csv')
holiday = pd.read_csv('drive/holiday.csv')

final_train.drop(['a'], axis=1, inplace=True)

# ! 修改时间格式
now_stamp = time.time()
local_time = datetime.datetime.fromtimestamp(now_stamp)
utc_time = datetime.datetime.utcfromtimestamp(now_stamp)
offset = local_time - utc_time


def utc2local(utc_str):
    # UTC时间转本地时间（+8:00
    UTC_FORMAT = '%Y-%m-%d %H:%M:%S'
    utc = datetime.datetime.strptime(utc_str, UTC_FORMAT)
    local = utc + offset
    local_str = local.strftime('%Y-%m-%d %H:%M:%S')
    return local_str


predict_final['utc_time'] = predict_final['utc_time'].astype(str)
predict_final['local_time'] = predict_final['utc_time'].apply(
    lambda x: utc2local(x))
predict_final['year'] = predict_final['local_time'].apply(
    lambda x: x[:4]).astype(int)
predict_final['month'] = predict_final['local_time'].apply(
    lambda x: x[5:7]).astype(int)
predict_final['day'] = predict_final['local_time'].apply(
    lambda x: x[8:10]).astype(int)
predict_final['hour'] = predict_final['local_time'].apply(
    lambda x: x[11:13]).astype(int)

predict_final['local_time'] = pd.to_datetime(predict_final['local_time'])
predict_final['date'] = predict_final['local_time'].apply(
    lambda x: x.strftime(format='%Y%m%d')).astype(int)
predict_final = pd.merge(predict_final, holiday, how='left', on='date')
predict_final['holiday'].fillna(1, inplace=True)
predict_final['holiday'] = predict_final['holiday'].astype(int)
predict_final['dayofweek'] = predict_final['local_time'].dt.dayofweek.astype(
    int)


final_train['utc_time'] = final_train['utc_time'].astype(str)
final_train['local_time'] = final_train['utc_time'].apply(
    lambda x: utc2local(x))
final_train['year'] = final_train['local_time'].apply(
    lambda x: x[:4]).astype(int)
final_train['month'] = final_train['local_time'].apply(
    lambda x: x[5:7]).astype(int)
final_train['day'] = final_train['local_time'].apply(
    lambda x: x[8:10]).astype(int)
final_train['hour'] = final_train['local_time'].apply(
    lambda x: x[11:13]).astype(int)

final_train['local_time'] = pd.to_datetime(final_train['local_time'])
final_train['date'] = final_train['local_time'].apply(
    lambda x: x.strftime(format='%Y%m%d')).astype(int)
final_train = pd.merge(final_train, holiday, how='left', on='date')
final_train['holiday'].fillna(1, inplace=True)
final_train['holiday'] = final_train['holiday'].astype(int)
final_train['dayofweek'] = final_train['local_time'].dt.dayofweek.astype(int)


station_name = final_train.groupby('stationId').agg('count').index.tolist()
station_list = []
for name in station_name:
    station_train = final_train[final_train['stationId'] == name]
    station_list.append(station_train)

station_name = predict_final.groupby('stationId').agg('count').index.tolist()
prediction_list = []
for name in station_name:
    station_train = predict_final[predict_final['stationId'] == name]
    prediction_list.append(station_train)


# ! 构造时间特征
final_list = []
name_list = ['PM2.5', 'PM10', 'O3']

for df in prediction_list:
    for name in name_list:
        df[name + '_' + '1'] = df[name].shift(1)
        df[name + '_' + '24'] = df[name].shift(24)
        df[name + '_' + '48'] = df[name].shift(48)
        df[name + '_' + '48_mean'] = df[name].rolling(window=48).mean()
        df[name + '_' + '48_mean'] = df[name + '_' + '48_mean'].shift(1)
        df[name + '_' + '24_mean'] = df[name].rolling(window=24).mean()
        df[name + '_' + '24_mean'] = df[name + '_' + '24_mean'].shift(1)
        df[name + '_' + '48_max'] = df[name].rolling(window=48).max()
        df[name + '_' + '48_max'] = df[name + '_' + '48_max'].shift(1)
        df[name + '_' + '24_max'] = df[name].rolling(window=24).max()
        df[name + '_' + '24_max'] = df[name + '_' + '24_max'].shift(1)
        df[name + '_' + '48_std'] = df[name].rolling(window=48).std()
        df[name + '_' + '48_std'] = df[name + '_' + '48_std'].shift(1)
        df[name + '_' + '24_std'] = df[name].rolling(window=24).std()
        df[name + '_' + '24_std'] = df[name + '_' + '24_std'].shift(1)
        df[name + '_' + '48_min'] = df[name].rolling(window=48).min()
        df[name + '_' + '48_min'] = df[name + '_' + '48_min'].shift(1)
        df[name + '_' + '24_min'] = df[name].rolling(window=24).min()
        df[name + '_' + '24_min'] = df[name + '_' + '24_min'].shift(1)
        df[name + '_' + '2'] = df[name].shift(2)
        df[name + '_' + '1_diff'] = df[name + '_' + '1'] - df[name + '_' + '2']
        df[name + '_' + '3'] = df[name].shift(3)
        df[name + '_' + '12'] = df[name].shift(12)
        df[name + '_' + '8'] = df[name].shift(8)
    final_list.append(df)


prediction_new = pd.DataFrame()

for df in final_list:
    prediction_new = pd.concat([prediction_new, df], axis=0)

prediction_new = prediction_new.sort_values('utc_time').reset_index()
del prediction_new['index']


new_list = []
name_list = ['PM2.5', 'PM10', 'O3']

for df in station_list:
    for name in name_list:
        df[name + '_' + '1'] = df[name].shift(1)
        df[name + '_' + '24'] = df[name].shift(24)
        df[name + '_' + '48'] = df[name].shift(48)
        df[name + '_' + '48_mean'] = df[name].rolling(window=48).mean()
        df[name + '_' + '48_mean'] = df[name + '_' + '48_mean'].shift(1)
        df[name + '_' + '24_mean'] = df[name].rolling(window=24).mean()
        df[name + '_' + '24_mean'] = df[name + '_' + '24_mean'].shift(1)
        df[name + '_' + '48_max'] = df[name].rolling(window=48).max()
        df[name + '_' + '48_max'] = df[name + '_' + '48_max'].shift(1)
        df[name + '_' + '24_max'] = df[name].rolling(window=24).max()
        df[name + '_' + '24_max'] = df[name + '_' + '24_max'].shift(1)
        df[name + '_' + '48_std'] = df[name].rolling(window=48).std()
        df[name + '_' + '48_std'] = df[name + '_' + '48_std'].shift(1)
        df[name + '_' + '24_std'] = df[name].rolling(window=24).std()
        df[name + '_' + '24_std'] = df[name + '_' + '24_std'].shift(1)
        df[name + '_' + '48_min'] = df[name].rolling(window=48).min()
        df[name + '_' + '48_min'] = df[name + '_' + '48_min'].shift(1)
        df[name + '_' + '24_min'] = df[name].rolling(window=24).min()
        df[name + '_' + '24_min'] = df[name + '_' + '24_min'].shift(1)
        df[name + '_' + '2'] = df[name].shift(2)
        df[name + '_' + '1_diff'] = df[name + '_' + '1'] - df[name + '_' + '2']
        df[name + '_' + '3'] = df[name].shift(3)
        df[name + '_' + '12'] = df[name].shift(12)
        df[name + '_' + '8'] = df[name].shift(8)
    new_list.append(df)


new_train = pd.DataFrame()

for df in new_list:
    new_train = pd.concat([new_train, df], axis=0)

new_train = new_train.sort_values('utc_time').reset_index()
del new_train['index']


train = new_train[:222985]
valid = new_train[222985:246505]
test = new_train[246505:]


train_X_PM25 = train[['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                      'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                      'dayofweek', 'PM2.5_1', 'PM2.5_24', 'PM2.5_48', 'PM2.5_48_mean',
                      'PM2.5_24_mean', 'PM2.5_48_max', 'PM2.5_24_max', 'PM2.5_48_std',
                      'PM2.5_24_std', 'PM2.5_48_min', 'PM2.5_24_min', 'PM2.5_2', 'PM2.5_1_diff',
                      'PM2.5_3', 'PM2.5_12', 'PM2.5_8', 'PM10_1', 'PM10_2', 'PM10_1_diff',
                      'O3_1', 'O3_2', 'O3_1_diff']]


train_X_PM10 = train[['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity', 'pressure',
                      'temperature', 'wind_direction', 'wind_speed', 'holiday', 'dayofweek', 'PM2.5_1',
                      'PM2.5_2', 'PM2.5_1_diff', 'PM10_1',
                      'PM10_24', 'PM10_48', 'PM10_48_mean', 'PM10_24_mean', 'PM10_48_max',
                      'PM10_24_max', 'PM10_48_std', 'PM10_24_std', 'PM10_48_min',
                      'PM10_24_min', 'PM10_2', 'PM10_1_diff',
                      'O3_1', 'O3_2',
                      'O3_1_diff']]

train_X_O3 = train[['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity', 'pressure',
                    'temperature', 'wind_direction', 'wind_speed', 'holiday', 'dayofweek', 'PM2.5_1',
                    'PM2.5_2', 'PM2.5_1_diff', 'PM10_1', 'PM10_2', 'PM10_1_diff',
                    'O3_1', 'O3_24', 'O3_48', 'O3_48_mean', 'O3_24_mean', 'O3_48_max',
                    'O3_24_max', 'O3_48_std', 'O3_24_std', 'O3_48_min', 'O3_24_min', 'O3_2',
                    'O3_1_diff', 'O3_3', 'O3_12', 'O3_8']]


valid_X_PM25 = valid[['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity', 'pressure',
                      'temperature', 'wind_direction', 'wind_speed', 'holiday', 'dayofweek', 'PM2.5_1', 'PM2.5_24', 'PM2.5_48',
                      'PM2.5_48_mean', 'PM2.5_24_mean', 'PM2.5_48_max', 'PM2.5_24_max',
                      'PM2.5_48_std', 'PM2.5_24_std', 'PM2.5_48_min', 'PM2.5_24_min',
                      'PM2.5_2', 'PM2.5_1_diff', 'PM2.5_3', 'PM2.5_12', 'PM2.5_8', 'PM10_1',
                      'PM10_2', 'PM10_1_diff', 'O3_1', 'O3_2', 'O3_1_diff']]


valid_X_PM10 = valid[['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity', 'pressure',
                      'temperature', 'wind_direction', 'wind_speed', 'holiday', 'dayofweek', 'PM2.5_1',
                      'PM2.5_2', 'PM2.5_1_diff', 'PM10_1',
                      'PM10_24', 'PM10_48', 'PM10_48_mean', 'PM10_24_mean', 'PM10_48_max',
                      'PM10_24_max', 'PM10_48_std', 'PM10_24_std', 'PM10_48_min',
                      'PM10_24_min', 'PM10_2', 'PM10_1_diff',
                      'O3_1', 'O3_2',
                      'O3_1_diff']]


valid_X_O3 = valid[['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity', 'pressure',
                    'temperature', 'wind_direction', 'wind_speed', 'holiday', 'dayofweek', 'PM2.5_1',
                    'PM2.5_2', 'PM2.5_1_diff', 'PM10_1', 'PM10_2', 'PM10_1_diff',
                    'O3_1', 'O3_24', 'O3_48', 'O3_48_mean', 'O3_24_mean', 'O3_48_max',
                    'O3_24_max', 'O3_48_std', 'O3_24_std', 'O3_48_min', 'O3_24_min', 'O3_2',
                    'O3_1_diff', 'O3_3', 'O3_12', 'O3_8']]


test_X_PM25 = test[['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity', 'pressure',
                    'temperature', 'wind_direction', 'wind_speed', 'holiday', 'dayofweek', 'PM2.5_1', 'PM2.5_24', 'PM2.5_48',
                    'PM2.5_48_mean', 'PM2.5_24_mean', 'PM2.5_48_max', 'PM2.5_24_max',
                    'PM2.5_48_std', 'PM2.5_24_std', 'PM2.5_48_min', 'PM2.5_24_min',
                    'PM2.5_2', 'PM2.5_1_diff', 'PM2.5_3', 'PM2.5_12', 'PM2.5_8', 'PM10_1',
                    'PM10_2', 'PM10_1_diff', 'O3_1', 'O3_2', 'O3_1_diff']]

test_X_PM10 = test[['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity', 'pressure',
                    'temperature', 'wind_direction', 'wind_speed', 'holiday', 'dayofweek', 'PM2.5_1',
                    'PM2.5_2', 'PM2.5_1_diff', 'PM10_1',
                    'PM10_24', 'PM10_48', 'PM10_48_mean', 'PM10_24_mean', 'PM10_48_max',
                    'PM10_24_max', 'PM10_48_std', 'PM10_24_std', 'PM10_48_min',
                    'PM10_24_min', 'PM10_2', 'PM10_1_diff',
                    'O3_1', 'O3_2',
                    'O3_1_diff']]


test_X_O3 = test[['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity', 'pressure',
                  'temperature', 'wind_direction', 'wind_speed', 'holiday', 'dayofweek', 'PM2.5_1',
                  'PM2.5_2', 'PM2.5_1_diff', 'PM10_1', 'PM10_2', 'PM10_1_diff',
                  'O3_1', 'O3_24', 'O3_48', 'O3_48_mean', 'O3_24_mean', 'O3_48_max',
                  'O3_24_max', 'O3_48_std', 'O3_24_std', 'O3_48_min', 'O3_24_min', 'O3_2',
                  'O3_1_diff', 'O3_3', 'O3_12', 'O3_8']]

train_pm25_y = train['PM2.5']
valid_pm25_y = valid['PM2.5']
test_pm25_y = test['PM2.5']
train_pm10_y = train['PM10']
valid_pm10_y = valid['PM10']
test_pm10_y = test['PM10']
train_o3_y = train['O3']
valid_o3_y = valid['O3']
test_o3_y = test['O3']


# ! Cross-Validation
data = new_train[:246505]
data = shuffle(data)
kf = KFold(n_splits=5)
train_list = []
test_list = []
for train, test in kf.split(data):
    train_list.append(train)
    test_list.append(test)

train_data_0 = data.drop([i for i in test_list[0]], axis=0)
test_data_0 = data.drop([i for i in train_list[0]], axis=0)
train_data_1 = data.drop([i for i in test_list[1]], axis=0)
test_data_1 = data.drop([i for i in train_list[1]], axis=0)
train_data_2 = data.drop([i for i in test_list[2]], axis=0)
test_data_2 = data.drop([i for i in train_list[2]], axis=0)
train_data_3 = data.drop([i for i in test_list[3]], axis=0)
test_data_3 = data.drop([i for i in train_list[3]], axis=0)
train_data_4 = data.drop([i for i in test_list[4]], axis=0)
test_data_4 = data.drop([i for i in train_list[4]], axis=0)

train_data = [train_data_0, train_data_1,
              train_data_2, train_data_3, train_data_4]
test_data = [test_data_0, test_data_1, test_data_2, test_data_3, test_data_4]


train_25_X = train_data[0][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                            'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                            'dayofweek', 'PM2.5_1', 'PM2.5_24', 'PM2.5_48', 'PM2.5_48_mean',
                            'PM2.5_24_mean', 'PM2.5_48_max', 'PM2.5_24_max', 'PM2.5_48_std',
                            'PM2.5_24_std', 'PM2.5_48_min', 'PM2.5_24_min', 'PM2.5_2',
                            'PM2.5_1_diff', 'PM2.5_3', 'PM2.5_12', 'PM2.5_8', 'PM10_1', 'PM10_2',
                            'PM10_1_diff', 'O3_1', 'O3_2', 'O3_1_diff']]
train_10_X = train_data[0][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                            'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                            'dayofweek', 'PM2.5_1', 'PM2.5_2', 'PM2.5_1_diff', 'PM10_1', 'PM10_24',
                            'PM10_48', 'PM10_48_mean', 'PM10_24_mean', 'PM10_48_max', 'PM10_24_max',
                            'PM10_48_std', 'PM10_24_std', 'PM10_48_min', 'PM10_24_min', 'PM10_2',
                            'PM10_1_diff', 'O3_1', 'O3_2', 'O3_1_diff']]
train_3_X = train_data[0][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                           'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                           'dayofweek', 'PM2.5_1', 'PM2.5_2', 'PM2.5_1_diff', 'PM10_1', 'PM10_2',
                           'PM10_1_diff', 'O3_1', 'O3_24', 'O3_48', 'O3_48_mean', 'O3_24_mean',
                           'O3_48_max', 'O3_24_max', 'O3_48_std', 'O3_24_std', 'O3_48_min',
                           'O3_24_min', 'O3_2', 'O3_1_diff', 'O3_3', 'O3_12', 'O3_8']]

test_25_X = test_data[0][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                          'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                          'dayofweek', 'PM2.5_1', 'PM2.5_24', 'PM2.5_48', 'PM2.5_48_mean',
                          'PM2.5_24_mean', 'PM2.5_48_max', 'PM2.5_24_max', 'PM2.5_48_std',
                          'PM2.5_24_std', 'PM2.5_48_min', 'PM2.5_24_min', 'PM2.5_2',
                          'PM2.5_1_diff', 'PM2.5_3', 'PM2.5_12', 'PM2.5_8', 'PM10_1', 'PM10_2',
                          'PM10_1_diff', 'O3_1', 'O3_2', 'O3_1_diff']]
test_10_X = test_data[0][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                          'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                          'dayofweek', 'PM2.5_1', 'PM2.5_2', 'PM2.5_1_diff', 'PM10_1', 'PM10_24',
                          'PM10_48', 'PM10_48_mean', 'PM10_24_mean', 'PM10_48_max', 'PM10_24_max',
                          'PM10_48_std', 'PM10_24_std', 'PM10_48_min', 'PM10_24_min', 'PM10_2',
                          'PM10_1_diff', 'O3_1', 'O3_2', 'O3_1_diff']]
test_3_X = test_data[0][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                         'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                         'dayofweek', 'PM2.5_1', 'PM2.5_2', 'PM2.5_1_diff', 'PM10_1', 'PM10_2',
                         'PM10_1_diff', 'O3_1', 'O3_24', 'O3_48', 'O3_48_mean', 'O3_24_mean',
                         'O3_48_max', 'O3_24_max', 'O3_48_std', 'O3_24_std', 'O3_48_min',
                         'O3_24_min', 'O3_2', 'O3_1_diff', 'O3_3', 'O3_12', 'O3_8']]

train_25_y = train_data[0]['PM2.5']
test_25_y = test_data[0]['PM2.5']
train_10_y = train_data[0]['PM10']
test_10_y = test_data[0]['PM10']
train_3_y = train_data[0]['O3']
test_3_y = test_data[0]['O3']


lgb_PM25_model_0 = lgb.LGBMRegressor(learning_rate=0.01, n_estimators=400, max_depth=4,
                                     reg_lambda=0.3, reg_alpha=0.2, random_state=66).fit(train_25_X, train_25_y)


lgb_PM10_model_0 = lgb.LGBMRegressor(learning_rate=0.01, n_estimators=300, max_depth=5,
                                     reg_lambda=0.1, reg_alpha=0.1, random_state=66).fit(train_10_X, train_10_y)


lgb_O3_model_0 = lgb.LGBMRegressor(learning_rate=0.01, n_estimators=500, max_depth=4,
                                   reg_lambda=0.2, reg_alpha=0.1,
                                   random_state=66).fit(train_3_X, train_3_y)


train_25_X = train_data[1][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                            'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                            'dayofweek', 'PM2.5_1', 'PM2.5_24', 'PM2.5_48', 'PM2.5_48_mean',
                            'PM2.5_24_mean', 'PM2.5_48_max', 'PM2.5_24_max', 'PM2.5_48_std',
                            'PM2.5_24_std', 'PM2.5_48_min', 'PM2.5_24_min', 'PM2.5_2',
                            'PM2.5_1_diff', 'PM2.5_3', 'PM2.5_12', 'PM2.5_8', 'PM10_1', 'PM10_2',
                            'PM10_1_diff', 'O3_1', 'O3_2', 'O3_1_diff']]
train_10_X = train_data[1][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                            'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                            'dayofweek', 'PM2.5_1', 'PM2.5_2', 'PM2.5_1_diff', 'PM10_1', 'PM10_24',
                            'PM10_48', 'PM10_48_mean', 'PM10_24_mean', 'PM10_48_max', 'PM10_24_max',
                            'PM10_48_std', 'PM10_24_std', 'PM10_48_min', 'PM10_24_min', 'PM10_2',
                            'PM10_1_diff', 'O3_1', 'O3_2', 'O3_1_diff']]
train_3_X = train_data[1][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                           'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                           'dayofweek', 'PM2.5_1', 'PM2.5_2', 'PM2.5_1_diff', 'PM10_1', 'PM10_2',
                           'PM10_1_diff', 'O3_1', 'O3_24', 'O3_48', 'O3_48_mean', 'O3_24_mean',
                           'O3_48_max', 'O3_24_max', 'O3_48_std', 'O3_24_std', 'O3_48_min',
                           'O3_24_min', 'O3_2', 'O3_1_diff', 'O3_3', 'O3_12', 'O3_8']]

test_25_X = test_data[1][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                          'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                          'dayofweek', 'PM2.5_1', 'PM2.5_24', 'PM2.5_48', 'PM2.5_48_mean',
                          'PM2.5_24_mean', 'PM2.5_48_max', 'PM2.5_24_max', 'PM2.5_48_std',
                          'PM2.5_24_std', 'PM2.5_48_min', 'PM2.5_24_min', 'PM2.5_2',
                          'PM2.5_1_diff', 'PM2.5_3', 'PM2.5_12', 'PM2.5_8', 'PM10_1', 'PM10_2',
                          'PM10_1_diff', 'O3_1', 'O3_2', 'O3_1_diff']]
test_10_X = test_data[1][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                          'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                          'dayofweek', 'PM2.5_1', 'PM2.5_2', 'PM2.5_1_diff', 'PM10_1', 'PM10_24',
                          'PM10_48', 'PM10_48_mean', 'PM10_24_mean', 'PM10_48_max', 'PM10_24_max',
                          'PM10_48_std', 'PM10_24_std', 'PM10_48_min', 'PM10_24_min', 'PM10_2',
                          'PM10_1_diff', 'O3_1', 'O3_2', 'O3_1_diff']]
test_3_X = test_data[1][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                         'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                         'dayofweek', 'PM2.5_1', 'PM2.5_2', 'PM2.5_1_diff', 'PM10_1', 'PM10_2',
                         'PM10_1_diff', 'O3_1', 'O3_24', 'O3_48', 'O3_48_mean', 'O3_24_mean',
                         'O3_48_max', 'O3_24_max', 'O3_48_std', 'O3_24_std', 'O3_48_min',
                         'O3_24_min', 'O3_2', 'O3_1_diff', 'O3_3', 'O3_12', 'O3_8']]

train_25_y = train_data[1]['PM2.5']
test_25_y = test_data[1]['PM2.5']
train_10_y = train_data[1]['PM10']
test_10_y = test_data[1]['PM10']
train_3_y = train_data[1]['O3']
test_3_y = test_data[1]['O3']


lgb_PM25_model_1 = lgb.LGBMRegressor(learning_rate=0.03, n_estimators=400, max_depth=4,
                                     reg_lambda=0.1, reg_alpha=0.2,
                                     random_state=66).fit(train_25_X, train_25_y)


lgb_PM10_model_1 = lgb.LGBMRegressor(learning_rate=0.03, n_estimators=400, max_depth=5,
                                     reg_lambda=0.05, reg_alpha=0.05, random_state=66).fit(train_10_X, train_10_y)

lgb_O3_model_1 = lgb.LGBMRegressor(learning_rate=0.05, n_estimators=150, max_depth=5,
                                   reg_lambda=0.1, reg_alpha=0.2,
                                   random_state=66).fit(train_3_X, train_3_y)


train_25_X = train_data[2][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                            'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                            'dayofweek', 'PM2.5_1', 'PM2.5_24', 'PM2.5_48', 'PM2.5_48_mean',
                            'PM2.5_24_mean', 'PM2.5_48_max', 'PM2.5_24_max', 'PM2.5_48_std',
                            'PM2.5_24_std', 'PM2.5_48_min', 'PM2.5_24_min', 'PM2.5_2',
                            'PM2.5_1_diff', 'PM2.5_3', 'PM2.5_12', 'PM2.5_8', 'PM10_1', 'PM10_2',
                            'PM10_1_diff', 'O3_1', 'O3_2', 'O3_1_diff']]
train_10_X = train_data[2][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                            'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                            'dayofweek', 'PM2.5_1', 'PM2.5_2', 'PM2.5_1_diff', 'PM10_1', 'PM10_24',
                            'PM10_48', 'PM10_48_mean', 'PM10_24_mean', 'PM10_48_max', 'PM10_24_max',
                            'PM10_48_std', 'PM10_24_std', 'PM10_48_min', 'PM10_24_min', 'PM10_2',
                            'PM10_1_diff', 'O3_1', 'O3_2', 'O3_1_diff']]
train_3_X = train_data[2][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                           'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                           'dayofweek', 'PM2.5_1', 'PM2.5_2', 'PM2.5_1_diff', 'PM10_1', 'PM10_2',
                           'PM10_1_diff', 'O3_1', 'O3_24', 'O3_48', 'O3_48_mean', 'O3_24_mean',
                           'O3_48_max', 'O3_24_max', 'O3_48_std', 'O3_24_std', 'O3_48_min',
                           'O3_24_min', 'O3_2', 'O3_1_diff', 'O3_3', 'O3_12', 'O3_8']]

test_25_X = test_data[2][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                          'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                          'dayofweek', 'PM2.5_1', 'PM2.5_24', 'PM2.5_48', 'PM2.5_48_mean',
                          'PM2.5_24_mean', 'PM2.5_48_max', 'PM2.5_24_max', 'PM2.5_48_std',
                          'PM2.5_24_std', 'PM2.5_48_min', 'PM2.5_24_min', 'PM2.5_2',
                          'PM2.5_1_diff', 'PM2.5_3', 'PM2.5_12', 'PM2.5_8', 'PM10_1', 'PM10_2',
                          'PM10_1_diff', 'O3_1', 'O3_2', 'O3_1_diff']]
test_10_X = test_data[2][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                          'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                          'dayofweek', 'PM2.5_1', 'PM2.5_2', 'PM2.5_1_diff', 'PM10_1', 'PM10_24',
                          'PM10_48', 'PM10_48_mean', 'PM10_24_mean', 'PM10_48_max', 'PM10_24_max',
                          'PM10_48_std', 'PM10_24_std', 'PM10_48_min', 'PM10_24_min', 'PM10_2',
                          'PM10_1_diff', 'O3_1', 'O3_2', 'O3_1_diff']]
test_3_X = test_data[2][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                         'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                         'dayofweek', 'PM2.5_1', 'PM2.5_2', 'PM2.5_1_diff', 'PM10_1', 'PM10_2',
                         'PM10_1_diff', 'O3_1', 'O3_24', 'O3_48', 'O3_48_mean', 'O3_24_mean',
                         'O3_48_max', 'O3_24_max', 'O3_48_std', 'O3_24_std', 'O3_48_min',
                         'O3_24_min', 'O3_2', 'O3_1_diff', 'O3_3', 'O3_12', 'O3_8']]

train_25_y = train_data[2]['PM2.5']
test_25_y = test_data[2]['PM2.5']
train_10_y = train_data[2]['PM10']
test_10_y = test_data[2]['PM10']
train_3_y = train_data[2]['O3']
test_3_y = test_data[2]['O3']


lgb_PM25_model_2 = lgb.LGBMRegressor(learning_rate=0.03, n_estimators=250, max_depth=4,
                                     reg_lambda=0.1, reg_alpha=0.02,
                                     random_state=66).fit(train_25_X, train_25_y)


lgb_PM10_model_2 = lgb.LGBMRegressor(learning_rate=0.03, n_estimators=600, max_depth=3,
                                     reg_lambda=0.05, reg_alpha=0.05,
                                     random_state=66).fit(train_10_X, train_10_y)


lgb_O3_model_2 = lgb.LGBMRegressor(learning_rate=0.1, n_estimators=150, max_depth=5,
                                   reg_lambda=0.1, reg_alpha=0.2,
                                   random_state=66).fit(train_3_X, train_3_y)


train_25_X = train_data[3][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                            'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                            'dayofweek', 'PM2.5_1', 'PM2.5_24', 'PM2.5_48', 'PM2.5_48_mean',
                            'PM2.5_24_mean', 'PM2.5_48_max', 'PM2.5_24_max', 'PM2.5_48_std',
                            'PM2.5_24_std', 'PM2.5_48_min', 'PM2.5_24_min', 'PM2.5_2',
                            'PM2.5_1_diff', 'PM2.5_3', 'PM2.5_12', 'PM2.5_8', 'PM10_1', 'PM10_2',
                            'PM10_1_diff', 'O3_1', 'O3_2', 'O3_1_diff']]
train_10_X = train_data[3][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                            'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                            'dayofweek', 'PM2.5_1', 'PM2.5_2', 'PM2.5_1_diff', 'PM10_1', 'PM10_24',
                            'PM10_48', 'PM10_48_mean', 'PM10_24_mean', 'PM10_48_max', 'PM10_24_max',
                            'PM10_48_std', 'PM10_24_std', 'PM10_48_min', 'PM10_24_min', 'PM10_2',
                            'PM10_1_diff', 'O3_1', 'O3_2', 'O3_1_diff']]
train_3_X = train_data[3][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                           'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                           'dayofweek', 'PM2.5_1', 'PM2.5_2', 'PM2.5_1_diff', 'PM10_1', 'PM10_2',
                           'PM10_1_diff', 'O3_1', 'O3_24', 'O3_48', 'O3_48_mean', 'O3_24_mean',
                           'O3_48_max', 'O3_24_max', 'O3_48_std', 'O3_24_std', 'O3_48_min',
                           'O3_24_min', 'O3_2', 'O3_1_diff', 'O3_3', 'O3_12', 'O3_8']]

test_25_X = test_data[3][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                          'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                          'dayofweek', 'PM2.5_1', 'PM2.5_24', 'PM2.5_48', 'PM2.5_48_mean',
                          'PM2.5_24_mean', 'PM2.5_48_max', 'PM2.5_24_max', 'PM2.5_48_std',
                          'PM2.5_24_std', 'PM2.5_48_min', 'PM2.5_24_min', 'PM2.5_2',
                          'PM2.5_1_diff', 'PM2.5_3', 'PM2.5_12', 'PM2.5_8', 'PM10_1', 'PM10_2',
                          'PM10_1_diff', 'O3_1', 'O3_2', 'O3_1_diff']]
test_10_X = test_data[3][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                          'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                          'dayofweek', 'PM2.5_1', 'PM2.5_2', 'PM2.5_1_diff', 'PM10_1', 'PM10_24',
                          'PM10_48', 'PM10_48_mean', 'PM10_24_mean', 'PM10_48_max', 'PM10_24_max',
                          'PM10_48_std', 'PM10_24_std', 'PM10_48_min', 'PM10_24_min', 'PM10_2',
                          'PM10_1_diff', 'O3_1', 'O3_2', 'O3_1_diff']]
test_3_X = test_data[3][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                         'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                         'dayofweek', 'PM2.5_1', 'PM2.5_2', 'PM2.5_1_diff', 'PM10_1', 'PM10_2',
                         'PM10_1_diff', 'O3_1', 'O3_24', 'O3_48', 'O3_48_mean', 'O3_24_mean',
                         'O3_48_max', 'O3_24_max', 'O3_48_std', 'O3_24_std', 'O3_48_min',
                         'O3_24_min', 'O3_2', 'O3_1_diff', 'O3_3', 'O3_12', 'O3_8']]

train_25_y = train_data[3]['PM2.5']
test_25_y = test_data[3]['PM2.5']
train_10_y = train_data[3]['PM10']
test_10_y = test_data[3]['PM10']
train_3_y = train_data[3]['O3']
test_3_y = test_data[3]['O3']


lgb_PM25_model_3 = lgb.LGBMRegressor(learning_rate=0.03, n_estimators=350, max_depth=3,
                                     reg_lambda=0.2, reg_alpha=0.2, random_state=66).fit(train_25_X, train_25_y)

lgb_PM10_model_3 = lgb.LGBMRegressor(learning_rate=0.03, n_estimators=600, max_depth=3,
                                     reg_lambda=0.1, reg_alpha=0.1,
                                     random_state=66).fit(train_10_X, train_10_y)

lgb_O3_model_3 = lgb.LGBMRegressor(learning_rate=0.07, n_estimators=350, max_depth=5,
                                   reg_lambda=0.1, reg_alpha=0.2,
                                   random_state=66).fit(train_3_X, train_3_y)


train_25_X = train_data[4][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                            'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                            'dayofweek', 'PM2.5_1', 'PM2.5_24', 'PM2.5_48', 'PM2.5_48_mean',
                            'PM2.5_24_mean', 'PM2.5_48_max', 'PM2.5_24_max', 'PM2.5_48_std',
                            'PM2.5_24_std', 'PM2.5_48_min', 'PM2.5_24_min', 'PM2.5_2',
                            'PM2.5_1_diff', 'PM2.5_3', 'PM2.5_12', 'PM2.5_8', 'PM10_1', 'PM10_2',
                            'PM10_1_diff', 'O3_1', 'O3_2', 'O3_1_diff']]
train_10_X = train_data[4][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                            'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                            'dayofweek', 'PM2.5_1', 'PM2.5_2', 'PM2.5_1_diff', 'PM10_1', 'PM10_24',
                            'PM10_48', 'PM10_48_mean', 'PM10_24_mean', 'PM10_48_max', 'PM10_24_max',
                            'PM10_48_std', 'PM10_24_std', 'PM10_48_min', 'PM10_24_min', 'PM10_2',
                            'PM10_1_diff', 'O3_1', 'O3_2', 'O3_1_diff']]
train_3_X = train_data[4][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                           'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                           'dayofweek', 'PM2.5_1', 'PM2.5_2', 'PM2.5_1_diff', 'PM10_1', 'PM10_2',
                           'PM10_1_diff', 'O3_1', 'O3_24', 'O3_48', 'O3_48_mean', 'O3_24_mean',
                           'O3_48_max', 'O3_24_max', 'O3_48_std', 'O3_24_std', 'O3_48_min',
                           'O3_24_min', 'O3_2', 'O3_1_diff', 'O3_3', 'O3_12', 'O3_8']]

test_25_X = test_data[4][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                          'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                          'dayofweek', 'PM2.5_1', 'PM2.5_24', 'PM2.5_48', 'PM2.5_48_mean',
                          'PM2.5_24_mean', 'PM2.5_48_max', 'PM2.5_24_max', 'PM2.5_48_std',
                          'PM2.5_24_std', 'PM2.5_48_min', 'PM2.5_24_min', 'PM2.5_2',
                          'PM2.5_1_diff', 'PM2.5_3', 'PM2.5_12', 'PM2.5_8', 'PM10_1', 'PM10_2',
                          'PM10_1_diff', 'O3_1', 'O3_2', 'O3_1_diff']]
test_10_X = test_data[4][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                          'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                          'dayofweek', 'PM2.5_1', 'PM2.5_2', 'PM2.5_1_diff', 'PM10_1', 'PM10_24',
                          'PM10_48', 'PM10_48_mean', 'PM10_24_mean', 'PM10_48_max', 'PM10_24_max',
                          'PM10_48_std', 'PM10_24_std', 'PM10_48_min', 'PM10_24_min', 'PM10_2',
                          'PM10_1_diff', 'O3_1', 'O3_2', 'O3_1_diff']]
test_3_X = test_data[4][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity',
                         'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday',
                         'dayofweek', 'PM2.5_1', 'PM2.5_2', 'PM2.5_1_diff', 'PM10_1', 'PM10_2',
                         'PM10_1_diff', 'O3_1', 'O3_24', 'O3_48', 'O3_48_mean', 'O3_24_mean',
                         'O3_48_max', 'O3_24_max', 'O3_48_std', 'O3_24_std', 'O3_48_min',
                         'O3_24_min', 'O3_2', 'O3_1_diff', 'O3_3', 'O3_12', 'O3_8']]

train_25_y = train_data[4]['PM2.5']
test_25_y = test_data[4]['PM2.5']
train_10_y = train_data[4]['PM10']
test_10_y = test_data[4]['PM10']
train_3_y = train_data[4]['O3']
test_3_y = test_data[4]['O3']


lgb_PM25_model_4 = lgb.LGBMRegressor(learning_rate=0.03, n_estimators=350, max_depth=3,
                                     reg_lambda=0.2, reg_alpha=0.1,
                                     random_state=66).fit(train_25_X, train_25_y)


lgb_PM10_model_4 = lgb.LGBMRegressor(learning_rate=0.05, n_estimators=600, max_depth=3,
                                     reg_lambda=0.1, reg_alpha=0.1,
                                     random_state=66).fit(train_10_X, train_10_y)

lgb_O3_model_4 = lgb.LGBMRegressor(learning_rate=0.03, n_estimators=350, max_depth=4,
                                   reg_lambda=0.1, reg_alpha=0.1,
                                   random_state=66).fit(train_3_X, train_3_y)


# ! 迭代预测未来48小时的PM2.5, PM10, O3
final_prediction = pd.DataFrame()
for station in station_name:
    station_train = prediction_new[prediction_new['stationId'] == station].sort_values('utc_time').reset_index()
    del station_train['index']

    for i in range(48):
        station_train['PM2.5_1'][7091+i] = station_train['PM2.5'][7090+i]
        station_train['O3_1'][7091+i] = station_train['O3'][7090+i]
        station_train['PM10_1'][7091+i] = station_train['PM10'][7090+i]
        station_train['O3_2'][7091+i] = station_train['O3'][7089+i]
        station_train['PM2.5_2'][7091+i] = station_train['PM2.5'][7089+i]
        station_train['PM10_2'][7091+i] = station_train['PM10'][7089+i]
        station_train['O3_1_diff'][7091+i] = station_train['O3_1'][7091+i] - station_train['O3_2'][7091+i]
        station_train['PM10_1_diff'][7091+i] = station_train['PM10_1'][7091+i] - station_train['PM10_2'][7091+i]
        station_train['PM2.5_1_diff'][7091+i] = station_train['PM2.5_1'][7091+i] - station_train['PM2.5_2'][7091+i]
        station_train['PM2.5_24'][7091+i] = station_train['PM2.5'][7067+i]
        station_train['PM10_24'][7091+i] = station_train['PM10'][7067+i]
        station_train['O3_24'][7091+i] = station_train['O3'][7067+i]
        station_train['PM2.5_48'][7091+i] = station_train['PM2.5'][7043+i]
        station_train['PM10_48'][7091+i] = station_train['PM10'][7043+i]
        station_train['O3_48'][7091+i] = station_train['O3'][7043+i]

        station_train['PM2.5_24_mean'][7091+i] = sum(station_train['PM2.5'][7067+i: 7091+i]) / 24
        station_train['PM2.5_48_mean'][7091+i] = sum(station_train['PM2.5'][7043+i: 7091+i]) / 48
        station_train['PM2.5_48_max'][7091+i] = max(station_train['PM2.5'][7043+i: 7091+i])
        station_train['PM2.5_24_max'][7091+i] = max(station_train['PM2.5'][7067+i: 7091+i])
        station_train['PM2.5_48_min'][7091+i] = min(station_train['PM2.5'][7043+i: 7091+i])
        station_train['PM2.5_24_min'][7091+i] = min(station_train['PM2.5'][7067+i: 7091+i])
        station_train['PM2.5_48_std'][7091+i] = np.std(station_train['PM2.5'][7043+i: 7091+i])
        station_train['PM2.5_24_std'][7091+i] = np.std(station_train['PM2.5'][7067+i: 7091+i])

        station_train['PM10_24_mean'][7091+i] = sum(station_train['PM10'][7067+i: 7091+i]) / 24
        station_train['PM10_48_mean'][7091+i] = sum(station_train['PM10'][7043+i: 7091+i]) / 48
        station_train['PM10_48_max'][7091+i] = max(station_train['PM10'][7043+i: 7091+i])
        station_train['PM10_24_max'][7091+i] = max(station_train['PM10'][7067+i: 7091+i])
        station_train['PM10_48_min'][7091+i] = min(station_train['PM10'][7043+i: 7091+i])
        station_train['PM10_24_min'][7091+i] = min(station_train['PM10'][7067+i: 7091+i])
        station_train['PM10_48_std'][7091+i] = np.std(station_train['PM10'][7043+i: 7091+i])
        station_train['PM10_24_std'][7091+i] = np.std(station_train['PM10'][7067+i: 7091+i])

        station_train['O3_24_mean'][7091+i] = sum(station_train['O3'][7067+i: 7091+i]) / 24
        station_train['O3_48_mean'][7091+i] = sum(station_train['O3'][7043+i: 7091+i]) / 48
        station_train['O3_48_max'][7091+i] = max(station_train['O3'][7043+i: 7091+i])
        station_train['O3_24_max'][7091+i] = max(station_train['O3'][7067+i: 7091+i])
        station_train['O3_48_min'][7091+i] = min(station_train['O3'][7043+i: 7091+i])
        station_train['O3_24_min'][7091+i] = min(station_train['O3'][7067+i: 7091+i])
        station_train['O3_48_std'][7091+i] = np.std(station_train['O3'][7043+i: 7091+i])
        station_train['O3_24_std'][7091+i] = np.std(station_train['O3'][7067+i: 7091+i])

        station_train['PM2.5'][7091+i] = (lgb_PM25_model_0.predict(station_train[7091+i:7092+i][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity', 'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday', 'dayofweek', 'PM2.5_1', 'PM2.5_24', 'PM2.5_48', 'PM2.5_48_mean', 'PM2.5_24_mean', 'PM2.5_48_max', 'PM2.5_24_max', 'PM2.5_48_std', 'PM2.5_24_std', 'PM2.5_48_min', 'PM2.5_24_min', 'PM2.5_2', 'PM2.5_1_diff', 'PM2.5_3', 'PM2.5_12', 'PM2.5_8', 'PM10_1', 'PM10_2', 'PM10_1_diff', 'O3_1', 'O3_2', 'O3_1_diff']]) + lgb_PM25_model_1.predict(station_train[7091+i:7092+i][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity', 'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday', 'dayofweek', 'PM2.5_1', 'PM2.5_24', 'PM2.5_48', 'PM2.5_48_mean', 'PM2.5_24_mean', 'PM2.5_48_max', 'PM2.5_24_max', 'PM2.5_48_std', 'PM2.5_24_std', 'PM2.5_48_min', 'PM2.5_24_min',
       'PM2.5_2', 'PM2.5_1_diff', 'PM2.5_3', 'PM2.5_12', 'PM2.5_8', 'PM10_1',
       'PM10_2', 'PM10_1_diff', 'O3_1', 'O3_2', 'O3_1_diff']]) + lgb_PM25_model_2.predict(station_train[7091+i:7092+i][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity', 'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday', 'dayofweek', 'PM2.5_1', 'PM2.5_24', 'PM2.5_48',
       'PM2.5_48_mean', 'PM2.5_24_mean', 'PM2.5_48_max', 'PM2.5_24_max',
       'PM2.5_48_std', 'PM2.5_24_std', 'PM2.5_48_min', 'PM2.5_24_min',
       'PM2.5_2', 'PM2.5_1_diff', 'PM2.5_3', 'PM2.5_12', 'PM2.5_8', 'PM10_1',
       'PM10_2', 'PM10_1_diff', 'O3_1', 'O3_2', 'O3_1_diff']]) + lgb_PM25_model_3.predict(station_train[7091+i:7092+i][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity', 'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday', 'dayofweek', 'PM2.5_1', 'PM2.5_24', 'PM2.5_48',
       'PM2.5_48_mean', 'PM2.5_24_mean', 'PM2.5_48_max', 'PM2.5_24_max',
       'PM2.5_48_std', 'PM2.5_24_std', 'PM2.5_48_min', 'PM2.5_24_min',
       'PM2.5_2', 'PM2.5_1_diff', 'PM2.5_3', 'PM2.5_12', 'PM2.5_8', 'PM10_1',
       'PM10_2', 'PM10_1_diff', 'O3_1', 'O3_2', 'O3_1_diff']]) + lgb_PM25_model_4.predict(station_train[7091+i:7092+i][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity', 'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday', 'dayofweek', 'PM2.5_1', 'PM2.5_24', 'PM2.5_48', 'PM2.5_48_mean', 'PM2.5_24_mean', 'PM2.5_48_max', 'PM2.5_24_max', 'PM2.5_48_std', 'PM2.5_24_std', 'PM2.5_48_min', 'PM2.5_24_min', 'PM2.5_2', 'PM2.5_1_diff', 'PM2.5_3', 'PM2.5_12', 'PM2.5_8', 'PM10_1', 'PM10_2', 'PM10_1_diff', 'O3_1', 'O3_2', 'O3_1_diff']])) / 5

        station_train['PM10'][7091+i] = (lgb_PM10_model_0.predict(station_train[7091+i:7092+i][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity', 'pressure', 'temperature', 'wind_direction', 'wind_speed', 'holiday', 'dayofweek', 'PM2.5_1',
       'PM2.5_2', 'PM2.5_1_diff', 'PM10_1',
       'PM10_24', 'PM10_48', 'PM10_48_mean', 'PM10_24_mean', 'PM10_48_max',
       'PM10_24_max', 'PM10_48_std', 'PM10_24_std', 'PM10_48_min',
       'PM10_24_min', 'PM10_2', 'PM10_1_diff',
       'O3_1', 'O3_2',
       'O3_1_diff']]) + lgb_PM10_model_1.predict(station_train[7091+i:7092+i][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity','pressure',
                 'temperature', 'wind_direction', 'wind_speed', 'holiday', 'dayofweek', 'PM2.5_1',
       'PM2.5_2', 'PM2.5_1_diff', 'PM10_1',
       'PM10_24', 'PM10_48', 'PM10_48_mean', 'PM10_24_mean', 'PM10_48_max',
       'PM10_24_max', 'PM10_48_std', 'PM10_24_std', 'PM10_48_min',
       'PM10_24_min', 'PM10_2', 'PM10_1_diff',
       'O3_1', 'O3_2',
       'O3_1_diff']]) + lgb_PM10_model_2.predict(station_train[7091+i:7092+i][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity','pressure',
                 'temperature', 'wind_direction', 'wind_speed', 'holiday', 'dayofweek', 'PM2.5_1',
       'PM2.5_2', 'PM2.5_1_diff', 'PM10_1',
       'PM10_24', 'PM10_48', 'PM10_48_mean', 'PM10_24_mean', 'PM10_48_max',
       'PM10_24_max', 'PM10_48_std', 'PM10_24_std', 'PM10_48_min',
       'PM10_24_min', 'PM10_2', 'PM10_1_diff',
       'O3_1', 'O3_2',
       'O3_1_diff']]) + lgb_PM10_model_3.predict(station_train[7091+i:7092+i][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity','pressure',
                 'temperature', 'wind_direction', 'wind_speed', 'holiday', 'dayofweek', 'PM2.5_1',
       'PM2.5_2', 'PM2.5_1_diff', 'PM10_1',
       'PM10_24', 'PM10_48', 'PM10_48_mean', 'PM10_24_mean', 'PM10_48_max',
       'PM10_24_max', 'PM10_48_std', 'PM10_24_std', 'PM10_48_min',
       'PM10_24_min', 'PM10_2', 'PM10_1_diff',
       'O3_1', 'O3_2',
       'O3_1_diff']]) + lgb_PM10_model_4.predict(station_train[7091+i:7092+i][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity','pressure',
                 'temperature', 'wind_direction', 'wind_speed', 'holiday', 'dayofweek', 'PM2.5_1',
       'PM2.5_2', 'PM2.5_1_diff', 'PM10_1',
       'PM10_24', 'PM10_48', 'PM10_48_mean', 'PM10_24_mean', 'PM10_48_max',
       'PM10_24_max', 'PM10_48_std', 'PM10_24_std', 'PM10_48_min',
       'PM10_24_min', 'PM10_2', 'PM10_1_diff',
       'O3_1', 'O3_2',
       'O3_1_diff']])) / 5

        station_train['O3'][7091+i] = (lgb_O3_model_0.predict(station_train[7091+i:7092+i][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity','pressure',
                 'temperature', 'wind_direction', 'wind_speed', 'holiday', 'dayofweek', 'PM2.5_1',
       'PM2.5_2', 'PM2.5_1_diff', 'PM10_1', 'PM10_2', 'PM10_1_diff',
       'O3_1', 'O3_24', 'O3_48', 'O3_48_mean', 'O3_24_mean', 'O3_48_max',
       'O3_24_max', 'O3_48_std', 'O3_24_std', 'O3_48_min', 'O3_24_min', 'O3_2',
       'O3_1_diff', 'O3_3', 'O3_12', 'O3_8']]) + lgb_O3_model_1.predict(station_train[7091+i:7092+i][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity','pressure',
                 'temperature', 'wind_direction', 'wind_speed', 'holiday', 'dayofweek', 'PM2.5_1',
       'PM2.5_2', 'PM2.5_1_diff', 'PM10_1', 'PM10_2', 'PM10_1_diff',
       'O3_1', 'O3_24', 'O3_48', 'O3_48_mean', 'O3_24_mean', 'O3_48_max',
       'O3_24_max', 'O3_48_std', 'O3_24_std', 'O3_48_min', 'O3_24_min', 'O3_2',
       'O3_1_diff', 'O3_3', 'O3_12', 'O3_8']]) + lgb_O3_model_2.predict(station_train[7091+i:7092+i][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity','pressure',
                 'temperature', 'wind_direction', 'wind_speed', 'holiday', 'dayofweek', 'PM2.5_1',
       'PM2.5_2', 'PM2.5_1_diff', 'PM10_1', 'PM10_2', 'PM10_1_diff',
       'O3_1', 'O3_24', 'O3_48', 'O3_48_mean', 'O3_24_mean', 'O3_48_max',
       'O3_24_max', 'O3_48_std', 'O3_24_std', 'O3_48_min', 'O3_24_min', 'O3_2',
       'O3_1_diff', 'O3_3', 'O3_12', 'O3_8']]) + lgb_O3_model_3.predict(station_train[7091+i:7092+i][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity','pressure',
                 'temperature', 'wind_direction', 'wind_speed', 'holiday', 'dayofweek', 'PM2.5_1',
       'PM2.5_2', 'PM2.5_1_diff', 'PM10_1', 'PM10_2', 'PM10_1_diff',
       'O3_1', 'O3_24', 'O3_48', 'O3_48_mean', 'O3_24_mean', 'O3_48_max',
       'O3_24_max', 'O3_48_std', 'O3_24_std', 'O3_48_min', 'O3_24_min', 'O3_2',
       'O3_1_diff', 'O3_3', 'O3_12', 'O3_8']]) + lgb_O3_model_4.predict(station_train[7091+i:7092+i][['longitude', 'latitude', 'year', 'month', 'day', 'hour', 'humidity','pressure',
                 'temperature', 'wind_direction', 'wind_speed', 'holiday', 'dayofweek', 'PM2.5_1',
       'PM2.5_2', 'PM2.5_1_diff', 'PM10_1', 'PM10_2', 'PM10_1_diff',
       'O3_1', 'O3_24', 'O3_48', 'O3_48_mean', 'O3_24_mean', 'O3_48_max',
       'O3_24_max', 'O3_48_std', 'O3_24_std', 'O3_48_min', 'O3_24_min', 'O3_2',
       'O3_1_diff', 'O3_3', 'O3_12', 'O3_8']])) / 5

    final_prediction = pd.concat([final_prediction, station_train], axis=0)


final_result = final_prediction.sort_values('utc_time').reset_index()
del final_result['index']
result = final_result[248185:]


# ! 绘制拟合曲线
fix_station = result[result['stationId'] == 'aotizhongxin_aq']
fix_station['local_time'] = pd.to_datetime(fix_station['local_time'])
fix_station = fix_station.set_index(['local_time'])
plt.title('aotizhongxin_O3', fontsize=15)
plt.xticks(rotation=45)
plt.xlabel('time', fontsize=10)
plt.ylabel('concentration', fontsize=10)
# plt.plot(fix_station['O3'], 'r-')
plt.plot(fix_station['O3'], 'b-')


# ! 生成最终submission.csv
result_submission = result[[
    'stationId', 'day', 'hour', 'PM2.5', 'PM10', 'O3']]
station_dic = {}
station_dic_reverse = {}
station_sbumission_list = ['dongsi_aq', 'tiantan_aq', 'guanyuan_aq', 'wanshouxigong_aq', 'aotizhongxin_aq', 'nongzhanguan_aq', 'wanliu_aq', 'beibuxinqu_aq',
                           'zhiwuyuan_aq', 'fengtaihuayuan_aq', 'yungang_aq', 'gucheng_aq', 'fangshan_aq', 'daxing_aq', 'yizhuang_aq', 'tongzhou_aq',
                           'shunyi_aq', 'pingchang_aq', 'mentougou_aq', 'pinggu_aq', 'huairou_aq', 'miyun_aq', 'yanqin_aq', 'dingling_aq', 'badaling_aq',
                           'miyunshuiku_aq', 'donggaocun_aq', 'yongledian_aq', 'yufa_aq', 'liulihe_aq', 'qianmen_aq', 'yongdingmennei_aq', 'xizhimenbei_aq',
                           'nansanhuan_aq', 'dongsihuan_aq']
for i in range(len(station_sbumission_list)):
    station_dic[station_sbumission_list[i]] = i
    station_dic_reverse[i] = station_sbumission_list[i]
result_submission['stationId'] = result_submission['stationId'].apply(
    lambda x: station_dic[x])


def rep_hour(day, hour):
    if day > 1:
        hour += 24
    return hour


result_submission['hour'] = result_submission.apply(
    lambda row: rep_hour(row['day'], row['hour']), axis=1)
result_submission['hour'] = result_submission['hour'].astype(int)
final_res_sub = result_submission.sort_values(by=['stationId', 'hour'])
final_res_sub['stationId'] = final_res_sub['stationId'].apply(
    lambda x: station_dic_reverse[x])


def rep_id(station_id, hour):
    hour = str(int(hour))
    res = station_id + '#' + hour
    return res


final_res_sub['stationId'] = final_res_sub.apply(
    lambda row: rep_id(row['stationId'], row['hour']), axis=1)
final_res_sub = final_res_sub.reset_index()
final_res_sub.drop(['index', 'day', 'hour'], axis=1, inplace=True)
final_res_sub.columns = ['test_id', 'PM2.5', 'PM10', 'O3']
final_res_sub.to_csv('drive/submission.csv', index=None)
