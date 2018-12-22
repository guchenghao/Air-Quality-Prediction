#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /Users/guchenghao/Desktop/5002Projcet/data_perprocessing.py
# Project: /Users/guchenghao/Desktop/5002Projcet
# Created Date: Wednesday, December 5th 2018, 2:59:57 pm
# Author: Harold Gu
# -----
# Last Modified: Wednesday, 5th December 2018 2:59:57 pm
# Modified By: Harold Gu
# -----
# Copyright (c) 2018 HKUST
# #
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import math
from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute, IterativeImputer
import seaborn as sns


# ! Read files
grid_weather = pd.read_csv('./grid_weather.csv')
air_quality = pd.read_csv('./air_quality.csv')
ob = pd.read_csv('observed_data.csv')

# ! construct completed time list
time_list = pd.Series(pd.date_range(
    start='2017-07-09', end='2018-05-03', freq='1H'))
time_list = pd.DataFrame(time_list)
time_list.columns = ['utc_time']
time_list = time_list[13:len(time_list)-1]


# ! construct distance matrix
grid_weather = grid_weather.drop_duplicates().reset_index()
grid_weather.drop(['index'], axis=1, inplace=True)

ob = ob.drop_duplicates().reset_index()
ob.drop(['index', 'Unnamed: 0'], axis=1, inplace=True)

air_quality = air_quality.drop_duplicates().reset_index()
air_quality.drop(['index'], axis=1, inplace=True)


grid_weather.drop(['Unnamed: 0'], axis=1, inplace=True)
grid_geo = grid_weather[['station_id',
                         'longitude', 'latitude']].drop_duplicates()

station_geo = air_quality[['stationId', 'longitude', 'latitude']]
station_geo = station_geo.drop_duplicates()
station_geo = station_geo.reset_index()
station_geo.drop(['index'], axis=1, inplace=True)

air_distances_station = []
for station in station_geo.values.tolist():
    longitudes_squre = np.power(
        np.array([station[1]]) - grid_geo['longitude'].values, 2)
    latitudes_squre = np.power(
        np.array([station[2]]) - grid_geo['latitude'].values, 2)
    distances = np.sqrt(longitudes_squre + latitudes_squre)
    air_distances_station.append(distances)

air_distances_station = np.array(air_distances_station)
air_distances_station = pd.DataFrame(
    air_distances_station, index=station_geo['stationId'].values, columns=grid_geo['station_id'].values)
air_distances_station = air_distances_station.T


# ! combine the grid weather info into air_quality
air_station = air_distances_station.columns
air = air_quality.copy()
tmp0 = pd.DataFrame()
filter_distance = []
for item in air_distances_station.columns.tolist():
    temp_distance = air_distances_station.sort_values(item)[:4][item]

    filter_distance.append(temp_distance)

filter_grid_weather_data = []
for item in filter_distance:
    grid_id_list = item.index.tolist()
    station_train = grid_weather.query(f'station_id in {str(grid_id_list)}')
    filter_grid_weather_data.append(station_train)


for i in range(len(air_station)):
    tmp = filter_grid_weather_data[i].groupby('time').agg('mean').reset_index()[['time', 'humidity', 'pressure',
                                                                                 'temperature', 'wind_direction', 'wind_speed']]
    tmp_station = pd.Series([air_station[i]] * len(tmp))

    tmp = pd.concat([tmp, tmp_station], axis=1)
    tmp = tmp.rename(columns={0: 'stationId',
                              'time': 'utc_time'})
    tmp0 = pd.concat([tmp0, tmp], axis=0)

air = pd.merge(air, tmp0, how='outer', on=['stationId', 'utc_time'])
air2 = air.sort_values('utc_time').reset_index()
del air2['index']
air2 = air2[159215:]


# ! define interpolate function
def weight_con(dataframe, index, forward, backward, item):
    weight1 = forward / (forward - backward)
    weight2 = -1 * backward / (forward - backward)
    value = dataframe[item][index - backward] * weight1 + \
        dataframe[item][index + forward] * weight2
    return value


# ! start to interpolate missing value
station_list = air2.groupby('stationId').agg('count').index
df_list = []
for station in station_list:
    tmp_df = air2[air2['stationId'] == station]
    tmp_df['utc_time'] = pd.to_datetime(tmp_df['utc_time'])
    # 与完整时间合并，保证时间连续，寻找缺失值
    tmp_df = pd.merge(tmp_df, time_list, how='outer', on='utc_time')
    tmp_df = tmp_df.sort_values('utc_time').reset_index()
    tmp_df['stationId'].fillna(tmp_df['stationId'][0], inplace=True)
    tmp_df.drop(['index', 'CO', 'SO2', 'NO2', 'year',
                 'month', 'day', 'hour'], axis=1, inplace=True)
    pm_na = tmp_df[tmp_df['PM2.5'].isna().values == True].drop_duplicates()
    o3_na = tmp_df[tmp_df['O3'].isna().values == True].drop_duplicates()
    pm10_na = tmp_df[tmp_df['PM10'].isna().values == True].drop_duplicates()

    for num in pm_na.index:
        forward = 1
        backward = -1
        if ((num + forward) in tmp_df.index.tolist()) and ((num + backward) in tmp_df.index.tolist()):
            while pd.isnull(tmp_df.iloc[num + forward]['PM2.5']) and forward <= 5:
                forward += 1
                if (num + forward) not in tmp_df.index.tolist():
                    break
            while pd.isnull(tmp_df.iloc[num + backward]['PM2.5']) and backward >= -5:
                backward -= 1
                if (num + backward) not in tmp_df.index.tolist():
                    break
            if (num + forward) not in tmp_df.index.tolist() or (num + backward) not in tmp_df.index.tolist():
                continue
            else:
                tmp_df['PM2.5'][num] = weight_con(
                    tmp_df, num, forward, backward, 'PM2.5')
        else:
            continue

    for num in o3_na.index:
        forward = 1
        backward = -1

        if ((num + forward) in tmp_df.index.tolist()) and ((num - backward) in tmp_df.index.tolist()):
            while pd.isnull(tmp_df.iloc[num + forward]['O3']) and forward <= 5:
                forward += 1
                if (num + forward) not in tmp_df.index.tolist():
                    break
            while pd.isnull(tmp_df.iloc[num + backward]['O3']) and backward >= -5:
                backward -= 1
                if (num + backward) not in tmp_df.index.tolist():
                    break
            if (num + forward) not in tmp_df.index.tolist() or (num + backward) not in tmp_df.index.tolist():
                continue
            else:
                tmp_df['O3'][num] = weight_con(
                    tmp_df, num, forward, backward, 'O3')
        else:
            continue

    for num in pm10_na.index:
        forward = 1
        backward = -1

        if ((num + forward) in tmp_df.index.tolist()) and ((num - backward) in tmp_df.index.tolist()):
            while pd.isnull(tmp_df.iloc[num + forward]['PM10']) and forward <= 5:
                forward += 1
                if (num + forward) not in tmp_df.index.tolist():
                    break
            while pd.isnull(tmp_df.iloc[num + backward]['PM10']) and backward >= -5:
                backward -= 1
                if (num + backward) not in tmp_df.index.tolist():
                    break
            if (num + forward) not in tmp_df.index.tolist() or (num + backward) not in tmp_df.index.tolist():
                continue
            else:
                tmp_df['PM10'][num] = weight_con(
                    tmp_df, num, forward, backward, 'PM10')
        else:
            continue

    df_list.append(tmp_df)
    print(len(df_list))


# ! fill NA Values by fancyimupte
final_fill_res = pd.DataFrame()
for item in df_list:
    item['stationId'].fillna(item['stationId'][0], inplace=True)
    item['utc_time'] = item['utc_time'].astype(object)
    item = item.sort_values('utc_time').reset_index()
    item.drop(['index'], axis=1, inplace=True)
    numerical_features = item.select_dtypes(exclude=["object"]).columns
    categorical_features = item.select_dtypes(include=["object"]).columns
    full_item = IterativeImputer().fit_transform(item[numerical_features])
#     full_item = KNN(k=5).fit_transform(item[numerical_features])
    full_item = pd.DataFrame(full_item, columns=numerical_features)
    full_item = pd.concat([item[categorical_features], full_item], axis=1)
    final_fill_res = pd.concat([final_fill_res, full_item], axis=0)


# ! drop some outliers
final_fill_res = final_fill_res.sort_values('utc_time').reset_index()
final_fill_res = final_fill_res[160405:]
final_fill_res.drop(['index', 'Unnamed: 0', 'CO',
                     'NO2', 'SO2'], axis=1, inplace=True)
final_fill_res = final_fill_res.drop_duplicates()
name_list = final_fill_res.groupby('stationId').agg('count').index
station_list = []
for name in name_list:
    df = final_fill_res[final_fill_res['stationId'] == name]
    station_list.append(df)

new_df = []
for df in station_list:
    for i in set(df['stationId']):
        if i == 'dingling_aq':
            df.drop([161463], axis=0, inplace=True)
        if i == 'donggaocun_aq':
            df.drop([199427], axis=0, inplace=True)
        if i == 'fengtaihuayuan_aq':
            df.drop([194117], axis=0, inplace=True)
        if i == 'gucheng_aq':
            df.drop([161791], axis=0, inplace=True)
        if i == 'mentougou_aq':
            df.drop([198600], axis=0, inplace=True)
        if i == 'qianmen_aq':
            df.drop([199429], axis=0, inplace=True)
        if i == 'xizhimenbei_aq':
            df.drop([199090], axis=0, inplace=True)
        if i == 'yongdingmennei_aq':
            df.drop([199445], axis=0, inplace=True)
    new_df.append(df)


# ! output training CSV file
final = pd.DataFrame()
for df in new_df:
    final = pd.concat([final, df], axis=0)

air2 = air2[-1700:]
air2 = air2.reset_index()
air2 = air2[:1680]
final.to_csv('./final.csv')
final = pd.concat([final, air2], axis=0)
final.drop(['CO', 'NO2', 'SO2'], axis=1, inplace=True)


final.drop(['day', 'hour', 'month', 'year', 'latitude',
            'longitude'], axis=1, inplace=True)
final = pd.merge(final, station_geo, how='left', on='stationId')
final = final.drop_duplicates()
final.drop(['index', 'Unnamed: 0'], axis=1, inplace=True)
final.to_csv('./prediction_final.csv')
