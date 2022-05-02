import os
import numpy as np
import pandas as pd
import math
import random
from datetime import datetime
import pickle
from utils import pkl_load, pad_nan_to_target
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import xarray as xr
def load_UCR(dataset):
    train_file = os.path.join('datasets/UCR', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('datasets/UCR', dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels
    
    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels


def load_UEA(dataset):
    train_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TEST.arff')[0]
    
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    
    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
    
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    
    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y
    
    
def load_forecast_npy(name, univar=False):
    data = np.load(f'datasets/{name}.npy')    
    if univar:
        data = data[: -1:]
        
    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)

    pred_lens = [24, 48, 96, 288, 672]
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, 0


def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float)

def _feature_transform(data):
    """
    原始特征对数变换
    """
    assert set(['open','high','low','close','avg_price','prev_close']) < set(data.columns), 'feature unsatisfied'
    data['return_rate'] = np.clip(data['close']/data['prev_close']-1, -0.2,0.2)  # 收益率标签
    data.iloc[:,:-1] = np.log(data.iloc[:,:-1]+1)
    diff_var = ['open','high','low','close','avg_price']
    data.loc[:,diff_var] = data.loc[:,diff_var].apply(lambda x:x-data['prev_close'])  # 差分
    return data

def _inverse_feature_transform(data):
    """
    _feature_transform对数变换反运算
    """
    assert set(['open','high','low','close','avg_price','prev_close']) < set(data.columns), 'feature unsatisfied'
    diff_var = ['open', 'high', 'low', 'close', 'avg_price']
    data.loc[:, diff_var] = data.loc[:, diff_var].apply(lambda x: x + data['prev_close'])  # 差分
    data.iloc[:,:-1] = np.exp(data.iloc[:,:-1]+1)
    return data


def load_forecast_parquet(file_path, log_trans=True):
    """
    load file then normalize it and date feature
    """
    # file_path = r'datasets/Astock_daily_2011_2022.h5'
    data = pd.read_parquet(file_path)
    if log_trans:
        data = _feature_transform(data)
    # 特征标准化
    scaler = StandardScaler().fit(data[:int(data.shape[0] * 0.6)])  # 计算zscore的scaler
    #     dt_scaler = StandardScaler().fit(dt_embed[:int(dt_embed_2d.shape[0]*0.6)])

    # 维度变换
    data = data.to_xarray().to_array(dim='factor')  # DataArray Type
    data = data.transpose('code', 'date', 'factor')  # 维度位置变换

    date_length = data.shape[1]  # 数据时间维度长度
    # 数据集划分
    train_slice = slice(None, int(0.6 * date_length))
    valid_slice = slice(int(0.6 * date_length), int(0.8 * date_length))
    test_slice = slice(int(0.8 * date_length), None)

    # 数据合并
    data_2dim = data.values.reshape(data.shape[0] * data.shape[1], data.shape[2])
    data_2dim = scaler.transform(data_2dim)  # 特征标准化
    data.values = data_2dim.reshape(data.shape[0], data.shape[1], data.shape[2])  # data数值做了标准化

    ## date process
    dt_embed = _get_time_features(pd.to_datetime(data.date.values))
    n_covariate_cols = dt_embed.shape[-1]  # 日期特征数量
    dt_scaler = StandardScaler().fit(dt_embed[train_slice])
    dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
    dt_embed = xr.DataArray(data=np.repeat(dt_embed, data.shape[0], axis=0),
                            dims=['code', 'date', 'factor'],
                            coords={
                                'code': data.code.values,
                                'date': data.date.values,
                                'factor': np.array(['dt_{}'.format(x) for x in range(dt_embed.shape[2])])
                            }
                            )  # dt_embed 三维DataArray数据
    data = xr.concat([dt_embed, data], dim='factor')  # 包含dt_embed的data
    pred_lens = [5, 10]  # 预测时序长度设置
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols


def load_forecast_csv(name, univar=False):
    data = pd.read_csv(f'datasets/{name}.csv', index_col='date', parse_dates=True)
    dt_embed = _get_time_features(data.index)  # 设置日期特征
    n_covariate_cols = dt_embed.shape[-1]  # 日期特征数量
    
    if univar:  # 对变量时间序列
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            data = data[['OT']]
        elif name == 'electricity':
            data = data[['MT_001']]
        else:
            data = data.iloc[:, -1:]  # 1 column dataframe
        
    data = data.to_numpy()
    if name == 'ETTh1' or name == 'ETTh2':
        train_slice = slice(None, 12*30*24)
        valid_slice = slice(12*30*24, 16*30*24)
        test_slice = slice(16*30*24, 20*30*24)
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12*30*24*4)
        valid_slice = slice(12*30*24*4, 16*30*24*4)
        test_slice = slice(16*30*24*4, 20*30*24*4)
    else:
        train_slice = slice(None, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    if name in ('electricity'):
        data = np.expand_dims(data.T, -1)  # Each variable is an instance rather than a feature
    else:
        data = np.expand_dims(data, 0)
    
    if n_covariate_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)
    
    if name in ('ETTh1', 'ETTh2', 'electricity'):
        pred_lens = [24, 48, 168, 336, 720]
    else:
        pred_lens = [24, 48, 96, 288, 672]
        
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols


def load_anomaly(name):
    res = pkl_load(f'datasets/{name}.pkl')
    return res['all_train_data'], res['all_train_labels'], res['all_train_timestamps'], \
           res['all_test_data'],  res['all_test_labels'],  res['all_test_timestamps'], \
           res['delay']


def gen_ano_train_data(all_train_data):
    maxl = np.max([ len(all_train_data[k]) for k in all_train_data ])
    pretrain_data = []
    for k in all_train_data:
        train_data = pad_nan_to_target(all_train_data[k], maxl, axis=0)
        pretrain_data.append(train_data)
    pretrain_data = np.expand_dims(np.stack(pretrain_data), 2)
    return pretrain_data
