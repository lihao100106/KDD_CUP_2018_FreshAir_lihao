#!/usr/bin/env python
# -*- coding: utf-8 -*-

# require package: chromedriver, lxml, selenium

from src.lihao.config import *
from lxml import etree
from selenium import webdriver
import pandas as pd
from result_merge import score_stacking


AQICN_PATH = "../output/aqicn_origin_data/"


def aqi2ug(cur_aqi):
    ug_bound = [0, 12, 35.4, 55.4, 150.4, 250.4, 350.4, 500.4]  # [0, 15, 40, 65, 150, 250, 350, 500]
    aqi_bound=[0, 50, 100, 150, 200, 300, 400, 500]
    for i, aqi in enumerate(aqi_bound):
        if cur_aqi <= aqi:
            break
    return ug_bound[i-1]+(cur_aqi-aqi_bound[i-1])*1.0/(aqi_bound[i]-aqi_bound[i-1])*(ug_bound[i]-ug_bound[i-1])


def aqi2ug_pm10(cur_aqi):
    ug_bound = [0, 54, 154, 254, 354, 424, 504, 604]
    aqi_bound=[0, 50, 100, 150, 200, 300, 400, 500]
    for i,aqi in enumerate(aqi_bound):
        if cur_aqi <= aqi:
            break
    return ug_bound[i-1]+(cur_aqi-aqi_bound[i-1])*1.0/(aqi_bound[i]-aqi_bound[i-1])*(ug_bound[i]-ug_bound[i-1])


def crawl_aqicn(city):
    driver = webdriver.Chrome()
    driver.get('http://aqicn.org/forecast/{}/'.format(city))

    page = driver.page_source
    driver.quit()

    result = []
    root = etree.HTML(page)
    tables = root.xpath('//table[@class="forecast-city-table"]')
    for table in tables:
        row = {}
        row['date'] = table.xpath('.//tr[@class="day"]/td/text()')[0]
        row['hours'] = table.xpath('.//tr[@class="hour"]/td/text()')
        row['aqi'] = []
        aqis = table.xpath('.//tr[@class="aqi"]/td')
        for aqi in aqis:
            row['aqi'].append(aqi.xpath('.//text()'))
        result.append(row)
    return result


def aqi_trans(result):
    aqicn_forecast = []
    for item in result:
        if item['date'].count(" ") > 0:
            day = int(item['date'].split(" ")[-1])
            for i in range(len(item['hours'])):
                aqicn_forecast.append({'day':day,'hour':item['hours'][i],'pm10':item['aqi'][i][0],'pm25':item['aqi'][i][1]})

    aqicn_forecast = pd.DataFrame(aqicn_forecast)
    aqicn_forecast['pm10'] = aqicn_forecast['pm10'].map(lambda x:aqi2ug_pm10(int(x)))
    aqicn_forecast['pm25'] = aqicn_forecast['pm25'].map(lambda x:aqi2ug(int(x)))
    return aqicn_forecast


def get_aqicn_data(submit_date):
    bj_result = crawl_aqicn('beijing')
    bj_forecast = aqi_trans(bj_result)
    print bj_forecast
    bj_forecast.to_csv(AQICN_PATH + "bj_aqi_data_{}.csv".format(submit_date), index=False)

    ld_result = crawl_aqicn('london')
    ld_forecast = aqi_trans(ld_result)
    print ld_forecast
    ld_forecast.to_csv(AQICN_PATH + "ld_aqi_data_{}.csv".format(submit_date), index=False)
    return


def merge_aqicn_data(date, origin_file):
    month_num = int(date.split("-")[1])
    day_num = int(date.split("-")[2])
    date_num_list = [day_num + 1, day_num + 2, day_num + 3]
    if month_num in [4, 6, 9, 11]:
        day_cnt = 30
    elif month_num in [1, 3, 5, 7, 8, 10, 12]:
        day_cnt = 31
    else:
        day_cnt = 29

    date_num_list = [(x - day_cnt) if x > day_cnt else x for x in date_num_list]
    print date, day_num, day_cnt, date_num_list

    bj_aqi = pd.read_table(AQICN_PATH + "bj_aqi_data_{}.csv".format(date), sep=",")
    bj_aqi = bj_aqi[bj_aqi['day'].isin(date_num_list)]
    bj_aqi['index'] = bj_aqi.apply(create_index, args=(day_num + 1, day_cnt, ), axis=1)
    bj_pm25 = dict(zip(bj_aqi['index'].tolist(), bj_aqi['pm25'].tolist()))
    bj_pm10 = dict(zip(bj_aqi['index'].tolist(), bj_aqi['pm10'].tolist()))
    print bj_aqi
    bj_pm25_li = bj_aq_interpolation(bj_pm25)
    bj_pm10_li = bj_aq_interpolation(bj_pm10)

    ld_aqi = pd.read_table(AQICN_PATH + "ld_aqi_data_{}.csv".format(date), sep=",")
    ld_aqi = ld_aqi[ld_aqi['day'].isin(date_num_list)]
    ld_aqi['index'] = ld_aqi.apply(create_index, args=(day_num + 1, day_cnt, ), axis=1)
    ld_pm25 = dict(zip(ld_aqi['index'].tolist(), ld_aqi['pm25'].tolist()))
    ld_pm10 = dict(zip(ld_aqi['index'].tolist(), ld_aqi['pm10'].tolist()))
    print ld_aqi
    ld_pm25_li = ld_aq_interpolation(ld_pm25)
    ld_pm10_li = ld_aq_interpolation(ld_pm10)

    # 8~23 0~23 0~7 beijing
    bj_pm25_list = []
    bj_pm25_li_list = []
    for i in range(48):
        index = (i + 8) / 3 * 3
        bj_pm25_list.append(bj_pm25.get(index))
        index = i + 8
        bj_pm25_li_list.append(bj_pm25_li.get(index))
    print "bj_pm25_list: {}".format(len(bj_pm25_list))
    print "bj_pm25_li_list: {}".format(len(bj_pm25_li_list))

    bj_pm10_list = []
    bj_pm10_li_list = []
    for i in range(48):
        index = (i + 8) / 3 * 3
        bj_pm10_list.append(bj_pm10.get(index))
        index = i + 8
        bj_pm10_li_list.append(bj_pm10_li.get(index))
    print "bj_pm10_list: {}".format(len(bj_pm10_list))
    print "bj_pm10_li_list: {}".format(len(bj_pm10_li_list))

    # 8~23 0~23 0~7 london
    ld_pm25_list = []
    ld_pm25_li_list = []
    for i in range(48):
        index = i / 3 * 3 + 8
        ld_pm25_list.append(ld_pm25.get(index))
        index = i + 8
        ld_pm25_li_list.append(ld_pm25_li.get(index))
    print "ld_pm25_list: {}".format(len(ld_pm25_list))
    print "ld_pm25_li_list: {}".format(len(ld_pm25_li_list))

    ld_pm10_list = []
    ld_pm10_li_list = []
    for i in range(48):
        index = i / 3 * 3 + 8
        if ld_pm10.get(index):
            ld_pm10_list.append(ld_pm10.get(index))
        index = i + 8
        if ld_pm10_li.get(index):
            ld_pm10_li_list.append(ld_pm10_li.get(index))
    print "ld_pm10_list: {}".format(len(ld_pm10_list))
    print "ld_pm10_li_list: {}".format(len(ld_pm10_li_list))

    file_name = origin_file.split("/")[-1].strip(".csv")
    file_path = "/".join(origin_file.split("/")[:-1]) + "/"

    # 使用线性插值后的aqicn数据
    df = pd.read_table(origin_file, sep=",")
    df['PM2.5_new'] = df.apply(replace_points_bj, args=("PM2.5", bj_pm25_li_list), axis=1)
    df['PM2.5_new'] = df.apply(replace_points_ld, args=("PM2.5_new", ld_pm25_li_list), axis=1)
    df['PM10_new'] = df.apply(replace_points_bj, args=("PM10", bj_pm10_li_list), axis=1)
    df['PM10_new'] = df.apply(replace_points_ld, args=("PM10_new", ld_pm10_li_list), axis=1)

    # 版本2: 替换
    df_v2 = df[['test_id', 'PM2.5_new', 'PM10_new', 'O3']]
    df_v2 = df_v2.rename(columns={'PM2.5_new': 'PM2.5', 'PM10_new': 'PM10'})
    df_v2.to_csv(file_path + "{}_r2.csv".format(file_name), index=False)

    # 版本5: 正式提交的版本 —— v4 采用动态调整权重的方式进行融合
    weight_dict = score_stacking(date)
    print weight_dict
    df = pd.read_table(origin_file, sep=",")
    df_bj = df.head(1680).copy()
    df_ld = df.tail(624).copy()
    df_bj['PM2.5_new'] = df_bj.apply(replace_points_bj, args=("PM2.5", bj_pm25_li_list), axis=1)
    df_bj['PM10_new'] = df_bj.apply(replace_points_bj, args=("PM10", bj_pm10_li_list), axis=1)
    df_bj["PM2.5_use"] = weight_dict['bj_PM2.5'][0] * df_bj["PM2.5"] + weight_dict['bj_PM2.5'][1] * df_bj['PM2.5_new']
    df_bj["PM10_use"] = weight_dict['bj_PM10'][0] * df_bj["PM10"] + weight_dict['bj_PM10'][1] * df_bj['PM10_new']
    df_bj_v5 = df_bj[['test_id', 'PM2.5_use', 'PM10_use', 'O3']]
    df_bj_v5 = df_bj_v5.rename(columns={'PM2.5_use': 'PM2.5', 'PM10_use': 'PM10'})

    df_ld['PM2.5_new'] = df_ld.apply(replace_points_ld, args=("PM2.5", ld_pm25_li_list), axis=1)
    df_ld['PM10_new'] = df_ld.apply(replace_points_ld, args=("PM10", ld_pm10_li_list), axis=1)
    df_ld["PM2.5_use"] = weight_dict['ld_PM2.5'][0] * df_ld["PM2.5"] + weight_dict['ld_PM2.5'][1] * df_ld['PM2.5_new']
    df_ld["PM10_use"] = weight_dict['ld_PM10'][0] * df_ld["PM10"] + weight_dict['ld_PM10'][1] * df_ld['PM10_new']
    df_ld_v5 = df_ld[['test_id', 'PM2.5_use', 'PM10_use', 'O3']]
    df_ld_v5 = df_ld_v5.rename(columns={'PM2.5_use': 'PM2.5', 'PM10_use': 'PM10'})
    df_v5 = pd.concat([df_bj_v5, df_ld_v5], axis=0)
    print df_v5.shape
    df_v5.to_csv(file_path + "{}_v4.csv".format(file_name), index=False)
    return


def bj_aq_interpolation(aq_dict):
    ret = dict()
    for key in range(0, max(aq_dict.keys())):
        begin = key / 3 * 3
        end = begin + 3
        if aq_dict.get(end):
            ret[key] = (aq_dict[end] - aq_dict[begin]) / 3.0 * (key - begin) + aq_dict[begin]
    return ret


def ld_aq_interpolation(aq_dict):
    ret = dict()
    for key in range(8, max(aq_dict.keys())):
        begin = (key - 2) / 3 * 3 + 2
        end = begin + 3
        if aq_dict.get(end):
            ret[key] = (aq_dict[end] - aq_dict[begin]) / 3.0 * (key - begin) + aq_dict[begin]
    return ret


def replace_points_bj(x, aq, aq_list):
    if str(x['test_id']).split("#")[0].count("_aq") > 0:
        index = int(str(x['test_id']).split("#")[1])
        return aq_list[index]
    else:
        return x[aq]


def replace_points_ld(x, aq, aq_list):
    if str(x['test_id']).split("#")[0].count("_aq") > 0:
        return x[aq]
    else:
        index = int(str(x['test_id']).split("#")[1])
        return aq_list[index]


def create_index(x, begin, day_cnt):
    if int(x['day']) - begin < 0:
        return 24 * (int(x['day']) - begin + day_cnt) + int(x['hour'])
    return 24 * (int(x['day']) - begin) + int(x['hour'])


if __name__ == "__main__":
    submit_date = SUBMIT_DATE
    # get_aqicn_data(submit_date)

    origin_file = "../output/result_lh/{}_lh.csv".format(submit_date)
    merge_aqicn_data(submit_date, origin_file)
