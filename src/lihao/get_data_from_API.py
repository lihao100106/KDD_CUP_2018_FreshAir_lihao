# coding:utf-8
import requests
import pandas as pd
import numpy as np
import os
from config import *


bj_path = 'data_beijing/'
ld_path = 'data_london/'
path = {'bj': bj_path, 'ld': ld_path}
bj_ret_path = bj_path + 'predict_data/predict_result_final/'
ld_ret_path = ld_path + 'predict_data/predict_result_final/'
api_path = {'bj': bj_path + 'api_data/', 'ld': ld_path + 'api_data/'}
date_path = {'bj': 'data_beijing/use_data/', 'ld': 'data_london/use_data/'}
ld_forecast = ['BL0', 'CD9', 'CD1', 'GN0', 'GR4', 'GN3', 'GR9', 'HV1', 'KF1', 'LW2', 'ST5', 'TH4', 'MY7']
ld_other = ['BX9', 'BX1', 'CT2', 'CT3', 'CR8', 'GB0', 'HR1', 'LH0', 'KC1', 'RB7', 'TD5']


def get_history_data(city, start, end, item, grid=None):
    item_dict = {"aq": "airquality", "meo": "meteorology"}
    item_use = item_dict.get(item)
    if grid and item_use == "meteorology":
        city_use = "{}_grid".format(city)
    else:
        city_use = city
    data_url = "https://biendata.com/competition/{item}/{city}/{start_time}/{end_time}/2k0d1d8".format(
        item=item_use,
        city=city_use,
        start_time=start,
        end_time=end)
    print data_url
    get_data = requests.get(url=data_url).text
    if grid is True:
        file_name = "{}_grid_{}.csv".format(end[:10], item)
    else:
        file_name = "{}_{}.csv".format(end[:10], item)
    save_file = open(api_path.get(city) + file_name, 'w')
    save_file.write(get_data)
    save_file.close()
    return


def merge_file_aq(city, date_list):
    df_list = []
    for date in date_list:
        df_s = pd.read_table(api_path.get(city) + "{}_aq.csv".format(date), sep=",", encoding="utf-8")
        df_list.append(df_s)
    df = pd.concat(df_list, axis=0)
    df = df.reset_index(drop=True)
    if 'id' in list(df):
        df = df.drop('id', axis=1)

    if city in ['bj', 'beijing']:
        df = df.rename(columns={
            "station_id": "stationId",
            "time": "utc_time",
            "PM25_Concentration": "PM2.5",
            "PM10_Concentration": "PM10",
            "NO2_Concentration": "NO2",
            "CO_Concentration": "CO",
            "O3_Concentration": "O3",
            "SO2_Concentration": "SO2",
        })
        df['key'] = df.apply(lambda x: "{}#{}".format(x['stationId'], x['utc_time']), axis=1)
        df = df.drop_duplicates(['key'], keep='last')
        df = df.drop('key', axis=1)
        df.sort_values(by=['stationId', 'utc_time'], ascending=True, inplace=True)
        # submit_date的23点数据手动剔除，仅在测试时开启！
        if IS_TEST:
            delete_time = ["{} 23:00:00".format(date_list[0])]
            df = df[~df['utc_time'].isin(delete_time)]
        df.to_csv(date_path.get(city) + "{}_aq.csv".format(date_list[0]), index=False, encoding="utf-8")
        return
    elif city in ['ld', 'london']:
        df = df.rename(columns={
            "station_id": "station_id",
            "time": "utc_time",
            "PM25_Concentration": "PM2.5",
            "PM10_Concentration": "PM10",
            "NO2_Concentration": "NO2",
        })
        df = df[['station_id', 'utc_time', 'PM2.5', 'PM10', 'NO2']]
        df['key'] = df.apply(lambda x: "{}#{}".format(x['station_id'], x['utc_time']), axis=1)
        df = df.drop_duplicates(['key'], keep='last')
        df = df.drop('key', axis=1)
        df.sort_values(by=['station_id', 'utc_time'], ascending=True, inplace=True)

        # submit_date的22点和23点数据手动剔除，仅在测试时开启！
        if IS_TEST:
            delete_time = ["{} 22:00:00".format(date_list[0]), "{} 23:00:00".format(date_list[0])]
            df = df[~df['utc_time'].isin(delete_time)]

        # 需要预测的
        df_fore = df[df['station_id'].isin(ld_forecast)]
        df_fore.to_csv(date_path.get(city) + "{}_aq_forecast.csv".format(date_list[0]), index=False, encoding="utf-8")

        # 不需要预测的
        df_other = df[df['station_id'].isin(ld_other)]
        df_other.to_csv(date_path.get(city) + "{}_aq_other.csv".format(date_list[0]), index=False, encoding="utf-8")
    else:
        print "CITY ERROR!"
    return


def merge_file_grid(city, date_list):
    name = {'bj': 'beijing', 'ld': 'london'}
    gps_info = pd.read_table(path.get(city) + "{}_grid_gps_info.csv".format(name.get(city)), sep=",", encoding="utf-8")

    df_list = []
    for date in date_list:
        df_s = pd.read_table(api_path.get(city) + "{}_grid_meo.csv".format(date), sep=",", encoding="utf-8")
        df_list.append(df_s)
    df = pd.concat(df_list, axis=0)
    df = df.reset_index(drop=True)
    df = df.rename(columns={
        "station_id": "stationName",
        "time": "utc_time",
        "wind_speed": "wind_speed/kph",
    })

    df = df.merge(gps_info, on='stationName', how='left')
    df = df.reset_index(drop=True)
    df['key'] = df.apply(lambda x: "{}#{}".format(x['stationName'], x['utc_time']), axis=1)
    df = df.drop_duplicates(['key'], keep='last')
    df = df.drop(['id', 'weather', 'key'], axis=1)
    df.sort_values(by=['stationName', 'utc_time'], ascending=True, inplace=True)
    df = df.ix[:, ['stationName', 'longitude', 'latitude', 'utc_time', 'temperature',
                   'pressure', 'humidity', 'wind_direction', 'wind_speed/kph']]
    df.to_csv(date_path.get(city) + "{}_grid_meo.csv".format(date_list[0]), index=False, encoding="utf-8")
    return


def merge_file_meo(date_list):
    city = 'bj'
    df_list = []
    for date in date_list:
        df_s = pd.read_table(api_path.get(city) + "{}_meo.csv".format(date), sep=",", encoding="utf-8")
        df_list.append(df_s)
    df = pd.concat(df_list, axis=0)
    df = df.reset_index(drop=True)
    if 'id' in list(df):
        df = df.drop('id', axis=1)
    df = df.rename(columns={"time": "utc_time"})
    df['key'] = df.apply(lambda x: "{}#{}".format(x['station_id'], x['utc_time']), axis=1)
    df = df.drop_duplicates(['key'], keep='last')
    df.sort_values(by=['station_id', 'utc_time'], ascending=True, inplace=True)
    df = df.ix[:, ['station_id', 'utc_time', 'temperature', 'pressure', 'humidity',
                   'wind_direction', 'wind_speed', 'weather']]
    df.to_csv(date_path.get(city) + "{}_meo.csv".format(date_list[0]), index=False, encoding="utf-8")
    return


def fill_na_with_predict(city, submit_day, df_time):
    begin = datetime.datetime.strptime(submit_day, "%Y-%m-%d")
    date_list = []
    for i in range(1, 5):
        date_list.append(str(begin - datetime.timedelta(days=i))[:10])

    if city in ['bj']:
        col_name = 'stationId'
        file_path = bj_ret_path
        name_pre = ''
    else:
        col_name = 'station_id'
        file_path = ld_ret_path
        name_pre = '_forecast'

    df_day_list = []
    predict_file_list = os.listdir(file_path)
    for date in date_list:
        file_name = "{}.csv".format(date)
        if file_name in predict_file_list:
            df_day = pd.read_table(file_path + file_name, sep=",")
            df_day_list.append(df_day)

    if len(df_day_list) > 0:
        predict = pd.concat(df_day_list, axis=0)
        predict.drop_duplicates(['key_col'], inplace=True, keep='first')

        # 读取原始aq文件
        df_aq = pd.read_table(date_path.get(city) + "{}_aq{}.csv".format(submit_day, name_pre), sep=",")
        df_list = []
        for station_id, df_station in df_aq.groupby(col_name):
            df_use = df_time.merge(df_station, on='utc_time', how='left')
            df_use[col_name] = station_id
            df_list.append(df_use)
        df_origin = pd.concat(df_list, axis=0)
        df_origin['key_col'] = df_origin.apply(create_key_col, args=(col_name, ), axis=1)
        df_modify = df_origin.merge(predict, on='key_col', how='left')

        aq_list = ['PM2.5', 'PM10']
        if city in ['bj']:
            aq_list.append('O3')
        for item in aq_list:
            df_modify['final_{}'.format(item)] = df_modify.apply(fill_na, args=(item,), axis=1)
        if city in ['bj']:
            chose_list = [col_name, 'utc_time', 'final_PM2.5', 'final_PM10', 'final_O3', 'NO2', 'CO', 'SO2']
            rename_info = {'final_PM2.5': 'PM2.5', 'final_PM10': 'PM10', 'final_O3': 'O3'}
        else:
            chose_list = [col_name, 'utc_time', 'final_PM2.5', 'final_PM10', 'NO2']
            rename_info = {'final_PM2.5': 'PM2.5', 'final_PM10': 'PM10'}
        df_final = df_modify[chose_list]
        df_final = df_final.rename(columns=rename_info)
        df_final.to_csv(date_path.get(city) + "{}_aq{}_filled.csv".format(submit_day, name_pre), index=False)
    return


def create_key_col(x, col):
    col_1 = x[col].split("_")[0]
    col_2 = x['utc_time']
    return "{}#{}".format(col_1, col_2)


def fill_na(x, item):
    if not np.isnan(x[item]):
        return x[item]
    if not np.isnan(x["predict_{}".format(item)]):
        return x["predict_{}".format(item)]
    return None


def data_update_master(city_list, submit_day):
    first_time = datetime.datetime.now()
    begin = datetime.datetime.strptime(submit_day, "%Y-%m-%d")
    start = '{}-00'.format(str(begin - datetime.timedelta(days=1))[:10])
    end = '{}-23'.format(str(begin)[:10])

    if 'bj' in city_list:
        get_history_data('bj', start, end, 'aq', grid=False)
        get_history_data('bj', start, end, 'meo', grid=True)
    if 'ld' in city_list:
        get_history_data('ld', start, end, 'aq', grid=False)
        get_history_data('ld', start, end, 'meo', grid=True)
    second_time = datetime.datetime.now()
    print "获取数据耗时 {:.2f} minutes".format((second_time - first_time).total_seconds() / 60.0)

    # 合并文件
    date_list = []
    for i in range(0, 4):
        date_list.append(str(begin - datetime.timedelta(days=i))[:10])
    if 'bj' in city_list:
        merge_file_aq('bj', date_list)
        merge_file_grid('bj', date_list)
    if 'ld' in city_list:
        merge_file_aq('ld', date_list)
        merge_file_grid('ld', date_list)

    # 利用历史预测值填充缺失值
    if BJ_FILL_NA or LD_FILL_NA:
        time_list = []
        for date in date_list:
            for i in range(24):
                each_time = datetime.datetime.strptime(date, "%Y-%m-%d") + datetime.timedelta(hours=i)
                time_list.append(str(each_time)[:19])
        df_time = pd.DataFrame(time_list, columns=['utc_time'])
        for city in city_list:
            try:
                fill_na_with_predict(city, submit_day, df_time)
            except:
                print "{} FILL aq NA ERROR!".format(city)
    return


def update_aq_file(start, end):
    # update beijing
    data_url = "https://biendata.com/competition/airquality/{city}/{start_time}/{end_time}/2k0d1d8".format(
        city='bj',
        start_time=start,
        end_time=end)
    print data_url
    get_data = requests.get(url=data_url).text
    data = [x.strip().split(",") for x in get_data.split('\n')]
    df_n = pd.DataFrame(data=data[1:], columns=data[0])
    print df_n.shape
    df_n = df_n.drop('id', axis=1)
    df_n = df_n.rename(columns={
        "station_id": "stationId",
        "time": "utc_time",
        "PM25_Concentration": "PM2.5",
        "PM10_Concentration": "PM10",
        "NO2_Concentration": "NO2",
        "CO_Concentration": "CO",
        "O3_Concentration": "O3",
        "SO2_Concentration": "SO2",
    })

    for open_file in ['201805']:
        df_o = pd.read_table("data_beijing/beijing_{}_aq.csv".format(open_file), sep=",")
        print df_o.shape
        df = pd.concat([df_o, df_n], axis=0)
        df = df.reset_index(drop=True)
        print df.shape
        df['key'] = df.apply(lambda x: "{}#{}".format(x['stationId'], x['utc_time']), axis=1)
        df.drop_duplicates(['key'], inplace=True, keep='last')
        df.sort_values(by=['stationId', 'utc_time'], ascending=True, inplace=True)
        df = df.drop('key', axis=1)
        print df.shape
        df.to_csv("data_beijing/beijing_{}_aq.csv".format(open_file), index=False)

    # update london
    data_url = "https://biendata.com/competition/airquality/{city}/{start_time}/{end_time}/2k0d1d8".format(
        city='ld',
        start_time=start,
        end_time=end)
    print data_url
    get_data = requests.get(url=data_url).text
    data = [x.strip().split(",") for x in get_data.split('\n')]
    df_n = pd.DataFrame(data=data[1:], columns=data[0])
    df_n = df_n.drop('id', axis=1)
    df_n = df_n.rename(columns={
        "station_id": "station_id",
        "time": "utc_time",
        "PM25_Concentration": "PM2.5",
        "PM10_Concentration": "PM10",
        "NO2_Concentration": "NO2",
    })
    df_n = df_n[['station_id', 'utc_time', 'PM2.5', 'PM10', 'NO2']]
    forecast_list = ['BL0', 'CD9', 'CD1', 'GN0', 'GR4', 'GN3', 'GR9', 'HV1', 'KF1', 'LW2', 'ST5', 'TH4', 'MY7']
    df_fore = df_n[df_n['station_id'].isin(forecast_list)]

    for open_file in ['201805']:
        df_need = pd.read_table("data_london/London_historical_aqi_forecast_{}.csv".format(open_file), sep=",")
        print df_need.shape
        df = pd.concat([df_need, df_fore], axis=0)
        df = df.reset_index(drop=True)
        print df.shape
        df['key'] = df.apply(lambda x: "{}#{}".format(x['station_id'], x['utc_time']), axis=1)
        df.drop_duplicates(['key'], inplace=True, keep='last')
        df.sort_values(by=['station_id', 'utc_time'], ascending=True, inplace=True)
        df = df.drop('key', axis=1)
        print df.shape
        df.to_csv("data_london/London_historical_aqi_forecast_{}.csv".format(open_file), index=False)
    return


if __name__ == '__main__':
    submit_date = SUBMIT_DATE
    begin_date = datetime.datetime.strptime(submit_date, "%Y-%m-%d") - datetime.timedelta(days=2)
    update_aq_file("{}-00".format(begin_date), "{}-23".format(submit_date))
