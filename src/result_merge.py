# coding:utf-8
import pandas as pd
import numpy as np
import operator
import simplejson as sjson
from src.lihao.utils import get_every_day, smape, create_key_col, create_key_col_v2
from src.lihao.config import *


aq_file_path = "lihao/"
result_path = "../output/"


def score_stacking(submit_day):
    end_day = datetime.datetime.strptime(submit_day, "%Y-%m-%d") - datetime.timedelta(days=1)
    begin_day = datetime.datetime.strptime(submit_day, "%Y-%m-%d") - datetime.timedelta(days=10)
    if begin_day < datetime.datetime.strptime("2018-05-01", "%Y-%m-%d"):
        begin_day = "2018-05-01"
    weight_dict = {
        "bj_PM2.5": [0.16, 0.69], "bj_PM10": [0.55, 0.29],
        "ld_PM2.5": [0.28, 0.82], "ld_PM10": [0.39, 0.30],
    }
    path = "output/result_lh/"
    for city in ['bj', 'ld']:
        for label in ['PM2.5', 'PM10']:
            if city == 'bj':
                bj_aq = pd.read_table(aq_file_path + "data_beijing/beijing_201805_aq.csv", sep=",")
                bj_aq = bj_aq[(bj_aq["PM2.5"] > 0) & (bj_aq["PM10"] > 0) & (bj_aq["O3"] > 0)]
                bj_aq["key_col"] = bj_aq.apply(create_key_col, args=("stationId",), axis=1)
            else:
                bj_aq = pd.read_table(aq_file_path + "data_london/London_historical_aqi_forecast_201805.csv", sep=",")
                bj_aq = bj_aq[(bj_aq["PM2.5"] > 0) & (bj_aq["PM10"] > 0)]
                bj_aq["key_col"] = bj_aq.apply(create_key_col, args=("station_id",), axis=1)

            true_list = []
            lh_list = []
            aqi_list = []
            for day in get_every_day(str(begin_day)[:10], str(end_day)[:10]):
                print day
                predict_begin = datetime.datetime.strptime(day, "%Y-%m-%d") + datetime.timedelta(days=1)
                df_lh = pd.read_table(path + "{}_lh.csv".format(day), sep=",")
                df_aqi = pd.read_table(path + "{}_lh_r2.csv".format(day), sep=",")
                if city == 'bj':
                    df_lh_bj = df_lh.head(1680).copy()
                    df_aqi_bj = df_aqi.head(1680).copy()
                else:
                    df_lh_bj = df_lh.tail(624).copy()
                    df_aqi_bj = df_aqi.tail(624).copy()

                df_lh_bj['utc_time'] = df_lh_bj['test_id'].apply(
                    lambda x: predict_begin + datetime.timedelta(hours=int(str(x).split("#")[1])))
                df_lh_bj['key_col'] = df_lh_bj.apply(create_key_col_v2, axis=1)

                df_aqi_bj['utc_time'] = df_aqi_bj['test_id'].apply(
                    lambda x: predict_begin + datetime.timedelta(hours=int(str(x).split("#")[1])))
                df_aqi_bj['key_col'] = df_aqi_bj.apply(create_key_col_v2, axis=1)

                df_lh_new = bj_aq.merge(df_lh_bj, on='key_col', how='inner')
                df_aqi_new = bj_aq.merge(df_aqi_bj, on='key_col', how='inner')
                lh_list += df_lh_new[label + '_y'].tolist()
                aqi_list += df_aqi_new[label + '_y'].tolist()
                true_list += df_lh_new[label + '_x'].tolist()
            true_np = np.array(true_list)
            lh_np = np.array(lh_list)
            aqi_np = np.array(aqi_list)
            print smape(true_list, lh_list), len(lh_list)
            print smape(true_list, aqi_list), len(aqi_list)
            stack_dict = dict()
            for x in np.arange(0.01, 1.001, 0.01):
                for y in np.arange(0.01, 1.001, 0.01):
                    key = "{:.2f},{:.2f}".format(x, y)
                    stack_dict[key] = smape(true_np, lh_np * x + aqi_np * y)
            ret = sorted(stack_dict.items(), key=operator.itemgetter(1), reverse=False)
            print "\n", city, label
            for k, v in ret[:5]:
                print k, v
            weight_dict["{}_{}".format(city, label)] = [float(x) for x in ret[0][0].split(",")]
    return weight_dict


def score_merge(submit_day, name_list, precision, use_day):
    time_1 = datetime.datetime.now()
    end_day = datetime.datetime.strptime(submit_day, "%Y-%m-%d") - datetime.timedelta(days=1)
    begin_day = datetime.datetime.strptime(submit_day, "%Y-%m-%d") - datetime.timedelta(days=use_day)
    if begin_day < datetime.datetime.strptime("2018-05-01", "%Y-%m-%d"):
        begin_day = "2018-05-01"
    weight_dict = {
        "bj_PM2.5": [0.0 for each in name_list], "bj_PM10": [0.0 for each in name_list],
        "ld_PM2.5": [0.0 for each in name_list], "ld_PM10": [0.0 for each in name_list],
    }

    bj_aq = pd.read_table(aq_file_path + "data_beijing/beijing_201805_aq.csv", sep=",")
    bj_aq = bj_aq[(bj_aq["PM2.5"] > 0) & (bj_aq["PM10"] > 0) & (bj_aq["O3"] > 0)]
    bj_aq["key_col"] = bj_aq.apply(create_key_col, args=("stationId",), axis=1)

    ld_aq = pd.read_table(aq_file_path + "data_london/London_historical_aqi_forecast_201805.csv", sep=",")
    ld_aq = ld_aq[(ld_aq["PM2.5"] > 0) & (ld_aq["PM10"] > 0)]
    ld_aq["key_col"] = ld_aq.apply(create_key_col, args=("station_id",), axis=1)
    df_id = pd.read_table(result_path + "test_id_demo.csv")
    for city in ['bj', 'ld']:
        for label in ['PM2.5', 'PM10', 'O3']:
            print "\n", city, label
            if city == 'bj':
                df_aq = bj_aq
            else:
                df_aq = ld_aq

            if city == 'ld' and label == 'O3':
                continue
            true_list = []
            lh_list = []
            sx_list = []
            jq_list = []
            zp_list = []
            aqi_list = []
            for day in get_every_day(str(begin_day)[:10], str(end_day)[:10]):
                if day in ["2018-04-26", "2018-04-27", "2018-04-28", "2018-04-29"]:
                    continue
                print day
                predict_begin = datetime.datetime.strptime(day, "%Y-%m-%d") + datetime.timedelta(days=1)
                df_lh = pd.read_table(result_path + "result_lh/{}_lh.csv".format(day), sep=",")
                df_sx = pd.read_table(result_path + "result_sx/{}_sx.csv".format(day), sep=",")
                df_jq = pd.read_table(result_path + "result_jq/{}_jq.csv".format(day), sep=",")
                df_zp = pd.read_table(result_path + "result_zp/{}_zp.csv".format(day), sep=",")
                df_jq = df_id.merge(df_jq, on='test_id', how='left')
                df_sx = df_id.merge(df_sx, on='test_id', how='left')
                df_zp = df_id.merge(df_zp, on='test_id', how='left')
                df_aqi = pd.read_table(result_path + "result_lh/{}_lh_r2.csv".format(day), sep=",")

                if city == 'bj':
                    df_lh_city = df_lh.head(1680).copy()
                    df_sx_city = df_sx.head(1680).copy()
                    df_jq_city = df_jq.head(1680).copy()
                    df_zp_city = df_zp.head(1680).copy()
                    df_aqi_city = df_aqi.head(1680).copy()
                else:
                    df_lh_city = df_lh.tail(624).copy()
                    df_sx_city = df_sx.tail(624).copy()
                    df_jq_city = df_jq.tail(624).copy()
                    df_zp_city = df_zp.tail(624).copy()
                    df_aqi_city = df_aqi.tail(624).copy()

                df_lh_city['utc_time'] = df_lh_city['test_id'].apply(
                    lambda x: predict_begin + datetime.timedelta(hours=int(str(x).split("#")[1])))
                df_lh_city['key_col'] = df_lh_city.apply(create_key_col_v2, axis=1)

                df_sx_city['utc_time'] = df_sx_city['test_id'].apply(
                    lambda x: predict_begin + datetime.timedelta(hours=int(str(x).split("#")[1])))
                df_sx_city['key_col'] = df_sx_city.apply(create_key_col_v2, axis=1)

                df_jq_city['utc_time'] = df_jq_city['test_id'].apply(
                    lambda x: predict_begin + datetime.timedelta(hours=int(str(x).split("#")[1])))
                df_jq_city['key_col'] = df_jq_city.apply(create_key_col_v2, axis=1)

                df_zp_city['utc_time'] = df_zp_city['test_id'].apply(
                    lambda x: predict_begin + datetime.timedelta(hours=int(str(x).split("#")[1])))
                df_zp_city['key_col'] = df_zp_city.apply(create_key_col_v2, axis=1)

                df_aqi_city['utc_time'] = df_aqi_city['test_id'].apply(
                    lambda x: predict_begin + datetime.timedelta(hours=int(str(x).split("#")[1])))
                df_aqi_city['key_col'] = df_aqi_city.apply(create_key_col_v2, axis=1)

                df_lh_new = df_aq.merge(df_lh_city, on='key_col', how='inner')
                df_sx_new = df_aq.merge(df_sx_city, on='key_col', how='inner')
                df_jq_new = df_aq.merge(df_jq_city, on='key_col', how='inner')
                df_zp_new = df_aq.merge(df_zp_city, on='key_col', how='inner')
                df_aqi_new = df_aq.merge(df_aqi_city, on='key_col', how='inner')

                lh_list += df_lh_new[label + '_y'].tolist()
                sx_list += df_sx_new[label + '_y'].tolist()
                jq_list += df_jq_new[label + '_y'].tolist()
                zp_list += df_zp_new[label + '_y'].tolist()
                aqi_list += df_aqi_new[label + '_y'].tolist()
                true_list += df_lh_new[label + '_x'].tolist()
            true_np = np.array(true_list)
            lh_np = np.array(lh_list)
            sx_np = np.array(sx_list)
            jq_np = np.array(jq_list)
            zp_np = np.array(zp_list)
            aqi_np = np.array(aqi_list)
            if 'lh' in name_list:
                print " lh:", smape(true_list, lh_list), len(lh_list)
            if 'sx' in name_list:
                print " sx:", smape(true_list, sx_list), len(sx_list)
            if 'jq' in name_list:
                print " jq:", smape(true_list, jq_list), len(jq_list)
            if 'zp' in name_list:
                print " zp:", smape(true_list, zp_list), len(zp_list)
            if 'aqi' in name_list:
                print "aqi:", smape(true_list, aqi_list), len(aqi_list)

            stack_dict = dict()
            cnt = 0
            if label == 'O3':
                if len(name_list) <= 2:
                    p = 0.01
                elif len(name_list) <= 4 and 'aqi' in name_list:
                    p = 0.02
                else:
                    p = 0.04
            else:
                p = precision

            if 'lh' in name_list:
                x_list = np.arange(0.00, 1.001, p)
            else:
                x_list = [0.0]

            if 'sx' in name_list:
                y_list = np.arange(0.00, 1.001, p)
            else:
                y_list = [0.0]

            if 'jq' in name_list:
                z_list = np.arange(0.00, 1.001, p)
            else:
                z_list = [0.0]

            if 'zp' in name_list:
                v_list = np.arange(0.00, 1.001, p)
            else:
                v_list = [0.0]

            if 'aqi' in name_list:
                if label == 'O3':
                    w_list = [0.0]
                else:
                    w_list = np.arange(0.00, 1.001, p)
            else:
                w_list = [0.0]

            print "loop cnt: {}".format(len(x_list) * len(y_list) * len(z_list) * len(v_list) * len(w_list))

            for x in x_list:
                for y in y_list:
                    for z in z_list:
                        for v in v_list:
                            for w in w_list:
                                cnt += 1
                                if cnt % 50000 == 0:
                                    print "calculate: {:7d}".format(cnt)
                                key = "{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(x, y, z, v, w)
                                merge_np = lh_np * x + sx_np * y + jq_np * z + zp_np * v + aqi_np * w
                                stack_dict[key] = smape(true_np, merge_np)
            ret = sorted(stack_dict.items(), key=operator.itemgetter(1), reverse=False)
            for k, v in ret[:5]:
                print k, v
            weight_dict["{}_{}".format(city, label)] = [float(x) for x in ret[0][0].split(",")]
    time_2 = datetime.datetime.now()
    print "use time: {:.2f} mints".format((time_2 - time_1).total_seconds() / 60.0)
    return weight_dict


def score_merge_master(begin, end, name_list, precision, use_day):
    df_id = pd.read_table(result_path + "test_id_demo.csv")
    for day in get_every_day(begin, end):
        print "submit date: {}".format(day)
        weight = score_merge(day, name_list, precision, use_day)
        weight_file = open(result_path + "score_merge_weight.txt", 'a')
        msg = "{}:{}".format(day, sjson.dumps(weight))
        print msg
        weight_file.write(msg + "\n")
        weight_file.close()
        df_lh = pd.read_table(result_path + "result_lh/{}_lh.csv".format(day), sep=",")
        df_lh_bj = df_lh.head(1680).copy()
        df_lh_ld = df_lh.tail(624).copy()

        df_sx = pd.read_table(result_path + "result_sx/{}_sx.csv".format(day), sep=",")
        df_sx = df_id.merge(df_sx, on='test_id', how='left')
        df_sx_bj = df_sx.head(1680).copy()
        df_sx_ld = df_sx.tail(624).copy()

        df_jq = pd.read_table(result_path + "result_jq/{}_jq.csv".format(day), sep=",")
        df_jq = df_id.merge(df_jq, on='test_id', how='left')
        df_jq_bj = df_jq.head(1680).copy()
        df_jq_ld = df_jq.tail(624).copy()

        if 'zp' in name_list:
            df_zp = pd.read_table(result_path + "result_zp/{}_zp.csv".format(day), sep=",")
            df_zp = df_id.merge(df_zp, on='test_id', how='left')
        else:
            df_zp = df_id.copy()
            for aq in ["PM2.5", "PM10", "O3"]:
                df_zp[aq] = 0
        df_zp_bj = df_zp.head(1680).copy()
        df_zp_ld = df_zp.tail(624).copy()

        df_aqi = pd.read_table(result_path + "result_lh/{}_lh_r2.csv".format(day), sep=",")
        df_aqi = df_aqi.fillna(0)
        df_aqi_bj = df_aqi.head(1680).copy()
        df_aqi_ld = df_aqi.tail(624).copy()

        df_bj = df_lh_bj[['test_id']].copy()
        for label in ["PM2.5", "PM10", "O3"]:
            df_bj[label] = weight['bj_{}'.format(label)][0] * df_lh_bj[label] + \
                           weight['bj_{}'.format(label)][1] * df_sx_bj[label] + \
                           weight['bj_{}'.format(label)][2] * df_jq_bj[label] + \
                           weight['bj_{}'.format(label)][3] * df_zp_bj[label] + \
                           weight['bj_{}'.format(label)][4] * df_aqi_bj[label]

        df_ld = df_lh_ld[['test_id']].copy()
        for label in ["PM2.5", "PM10"]:
            df_ld[label] = weight['ld_{}'.format(label)][0] * df_lh_ld[label] + \
                           weight['ld_{}'.format(label)][1] * df_sx_ld[label] + \
                           weight['ld_{}'.format(label)][2] * df_jq_ld[label] + \
                           weight['ld_{}'.format(label)][3] * df_zp_ld[label] + \
                           weight['ld_{}'.format(label)][4] * df_aqi_ld[label]

        df_merge = pd.concat([df_bj, df_ld], axis=0)
        df_merge = df_merge.ix[:, ["test_id", "PM2.5", "PM10", "O3"]]
        if use_day == 10:
            save_name = "{}_merge_{}.csv".format(day, "+".join(name_list))
        else:
            save_name = "{}_merge_{}_{}day.csv".format(day, "+".join(name_list), use_day)
        df_merge.to_csv(result_path + "result_merge/" + save_name, index=False)
    return


if __name__ == '__main__':
    begin_date = SUBMIT_DATE
    end_date = SUBMIT_DATE
    for use_days in [5, 7, 10]:
        print "使用过去{}天的预测结果训练权重".format(use_days)
        score_merge_master(
            begin=begin_date,
            end=end_date,
            name_list=['lh', 'sx', 'jq', 'zp', 'aqi'],
            precision=0.04,
            use_day=use_days)
