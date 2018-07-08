# coding:utf-8
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt, degrees, atan2
from config import *


def smape(truth, pred):
    if isinstance(truth, (list, np.ndarray, pd.Series)):
        truth = np.array(truth, dtype=float)
        pred = np.maximum(np.array(pred, dtype=float), 0)
        return (np.abs(truth - pred)/(np.maximum(truth + pred,  0.000001)/2)).mean()
    else:  # 认为是数值
        pred = float(max(pred, 0))
        return abs(pred - truth) / max((pred + truth) / 2, 0.000001)


def haversine(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r


def direction_degree(lonA, latA, lonB, latB):
    # lon1 -> lon2
    radLatA = radians(latA)
    radLonA = radians(lonA)
    radLatB = radians(latB)
    radLonB = radians(lonB)
    dLon = radLonB - radLonA
    y = sin(dLon) * cos(radLatB)
    x = cos(radLatA) * sin(radLatB) - sin(radLatA) * cos(radLatB) * cos(dLon)
    brng = degrees(atan2(y, x))
    brng = (brng + 360) % 360
    return brng


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return "smape", smape(labels, preds)


def get_every_day(begin_date, end_date):
    date_list = []
    begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    while begin_date <= end_date:
        date_str = begin_date.strftime("%Y-%m-%d")
        date_list.append(date_str)
        begin_date += datetime.timedelta(days=1)
    return date_list


def score_fit(df_x, df_y, x, y, item):
    if (x + y) > 1.000001 or (x + y) < 0.999999:
        print "x+y必须等于1"
        return
    if df_x.shape[0] >= df_y.shape[0]:
        df_keep = df_x[~df_x['key_col'].isin(df_y['key_col'].tolist())]
        df_inner = df_x.merge(df_y, on='key_col', how='inner')
        for aq in item:
            df_inner[aq] = x * df_inner["{}_x".format(aq)] + y * df_inner["{}_y".format(aq)]
        df_inner = df_inner.ix[:, ['key_col'] + item]
        ret_df = pd.concat([df_keep, df_inner], axis=0)
        ret_df = ret_df.sort_values(by="key_col", ascending=True)
        ret_df = ret_df.reset_index(drop=True)
        return ret_df
    else:
        return score_fit(df_y, df_x, y, x)


def cal_aq_by_min(x, item, min_dict):
    s_id = x['key_col'].split("#")[0]
    if x["predict_{}".format(item)] <= min_dict[s_id][item]:
        return min_dict[s_id][item]
    return x["predict_{}".format(item)]


def create_key_col(x, col):
    ret = "{}#{}".format(x[col], str(x['utc_time'])[:19])
    return ret


def create_key_col_v2(x):
    ret = "{}#{}".format(x['test_id'].split("#")[0], str(x['utc_time'])[:19])
    return ret

def bj_std_col(x, time_dict):
    col_1 = x['key_col'].split("#")[0] + '_aq'
    col_2 = time_dict.get(x['key_col'].split("#")[1])
    return "{}#{}".format(col_1, col_2)


def ld_std_col(x, time_dict):
    col_1 = x['key_col'].split("#")[0]
    col_2 = time_dict.get(x['key_col'].split("#")[1])
    return "{}#{}".format(col_1, col_2)


def lh_result_merge(submit_date):
    predict_begin = datetime.datetime.strptime(submit_date, "%Y-%m-%d") + datetime.timedelta(hours=24)
    time_dict = dict()
    for i in range(48):
        each_time = str(predict_begin + datetime.timedelta(hours=i))[:19]
        time_dict[each_time] = i

    df_bj = pd.read_table("data_beijing/predict_data/predict_result_final/{}.csv".format(submit_date), sep=",")
    df_ld = pd.read_table("data_london/predict_data/predict_result_final/{}.csv".format(submit_date), sep=",")

    df_bj['test_id'] = df_bj.apply(bj_std_col, args=(time_dict,), axis=1)
    df_ld['test_id'] = df_ld.apply(ld_std_col, args=(time_dict,), axis=1)
    ret = pd.concat([df_bj, df_ld], axis=0)
    ret = ret.rename(columns={"predict_PM2.5": "PM2.5", "predict_PM10": "PM10", "predict_O3": "O3"})
    ret = ret[["test_id", "PM2.5", "PM10", "O3"]]
    if IS_TEST:
        save_name = "test/result_for_submit_{}.csv".format(submit_date)
    else:
        save_name = RESULT_NAME
    ret.to_csv(RESULT_PATH + save_name, index=False)
    print "save file done: {}".format(save_name)
    return
