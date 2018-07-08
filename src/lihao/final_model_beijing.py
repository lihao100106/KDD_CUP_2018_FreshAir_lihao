# coding=utf-8
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from extract_feature_beijing import FeatureCreate
from utils import get_every_day, score_fit, cal_aq_by_min
from config import *
import cPickle


weight = {"M": 4, "AD": 6}


class PredictBeijingAqByLIHAO(object):
    def __init__(self):
        self.submit_begin = "2018-04-01"
        self.submit_end = "2018-04-01"
        self.path = "data_beijing/predict_data/"
        self.result_path = "data_beijing/predict_data/predict_result_final/"
        self.label_dict = {"PM25": "PM2.5", "PM10": "PM10", "O3": "O3"}
        self.delay_hour = 0

    def predict_no_gap(self, df_feats, hours, chose_label, submit_day):
        # load all model
        label_col = self.label_dict.get(chose_label)
        model_list = []
        for model_id in range(hours):
            model_path = "model_beijing/"
            name_suf = ""
            model_file = model_path + "{}_BJ_XGB_PERIOD_24_LABEL_{}{}.pkl".format(chose_label, model_id + 1, name_suf)
            model_xgb = cPickle.load(open(model_file, 'rb'))
            model_list.append(model_xgb)

        predict_begin = datetime.datetime.strptime(submit_day, "%Y-%m-%d") + datetime.timedelta(hours=24)
        range_start = str(predict_begin - datetime.timedelta(hours=24+self.delay_hour))[:19]
        range_end = str(predict_begin - datetime.timedelta(hours=1+self.delay_hour))[:19]
        time_range = "{}#{}".format(range_start, range_end)
        time_range_list = [time_range]

        df_feats_use = df_feats[df_feats['time_range'].isin(time_range_list)]
        df_feats_use = df_feats_use.reset_index(drop=True)

        predict_np = np.zeros((df_feats_use.shape[0], hours))
        result_np = np.zeros((df_feats_use.shape[0], hours))
        for model_pkl in model_list:
            model_xgb = model_pkl['estimator']
            feats_list = model_pkl['feat_names']
            feat_use = xgb.DMatrix(df_feats_use.ix[:, feats_list].get_values())
            xgb_predict = model_xgb.predict(feat_use)
            ix = model_pkl['label_hours'] - 1
            predict_np[:, ix] = xgb_predict
        predict_np = np.maximum(predict_np, 0)
        result_np[:, 0] = predict_np[:, 0]
        for ix in range(1, hours):
            result_np[:, ix] = predict_np[:, ix] - predict_np[:, ix - 1]
        result_np = np.maximum(result_np, 0)

        df_result = df_feats_use[['time_range', 'station_id']]
        for ix in range(hours):
            df_result["{}h".format(ix + 1)] = result_np[:, ix]
        ret_list = []
        for index in range(df_result.shape[0]):
            line_dict = df_result.loc[index, :].to_dict()
            for i in range(1, hours + 1):
                predict_time = pd.to_datetime(line_dict['time_range'].split("#")[1]) + datetime.timedelta(
                    hours=i+self.delay_hour)
                key_col = "{}#{}".format(line_dict['station_id'], predict_time)
                ret_dict = dict()
                ret_dict["key_col"] = key_col
                ret_dict["predict_{}".format(label_col)] =  line_dict["{}h".format(i)]
                ret_list.append(ret_dict)
        predict_info = pd.DataFrame(ret_list)
        predict_info = predict_info.sort_values(by="key_col", ascending=True)
        predict_info = predict_info.reset_index(drop=True)
        return predict_info

    def predict_with_gap(self, df_feats, chose_label, submit_day):
        label_col = self.label_dict.get(chose_label)
        predict_origin = dict()
        predict_begin = datetime.datetime.strptime(submit_day, "%Y-%m-%d") + datetime.timedelta(hours=24)
        for time_gap in [12, 24, 36, 48]:
            time_range_list = []
            for i in range(time_gap):
                each_time = predict_begin + datetime.timedelta(hours=i)
                range_start = str(each_time - datetime.timedelta(hours=(time_gap + 23 + self.delay_hour)))[:19]
                range_end = str(each_time - datetime.timedelta(hours=time_gap + self.delay_hour))[:19]
                time_range_list.append("{}#{}".format(range_start, range_end))

            df_feats_use = df_feats[df_feats['time_range'].isin(time_range_list)]
            df_feats_use = df_feats_use.reset_index(drop=True)

            # load model
            model_path = "model_beijing/model_with_gap/"
            name_suf = ""
            model_file = model_path + "{}_BJ_XGB_PERIOD_24_GAP_{}{}.pkl".format(chose_label, time_gap, name_suf)
            model_dict = cPickle.load(open(model_file, 'rb'))
            model_xgb = model_dict['estimator']
            feats_list = model_dict['feat_names']

            feat_for_predict = xgb.DMatrix(df_feats_use.ix[:, feats_list].get_values())
            xgb_predict = model_xgb.predict(feat_for_predict)
            predict_np = np.maximum(xgb_predict, 0)

            df_result = df_feats_use[['time_range', 'station_id']]
            df_result["predict"] = predict_np
            ret_list = []
            for index in range(df_result.shape[0]):
                line_dict = df_result.loc[index, :].to_dict()
                predict_time = pd.to_datetime(line_dict['time_range'].split("#")[1]) + datetime.timedelta(
                    hours=time_gap+ self.delay_hour)
                key_col = "{}#{}".format(line_dict['station_id'], predict_time)
                ret_dict = dict()
                ret_dict["key_col"] = key_col
                ret_dict["predict_{}".format(label_col)] = line_dict["predict"]
                ret_list.append(ret_dict)
            predict_info = pd.DataFrame(ret_list)
            predict_info = predict_info.sort_values(by="key_col", ascending=True)
            predict_origin[time_gap] = predict_info

        df_ret = predict_origin.get(48)
        df_ret = score_fit(df_ret, predict_origin.get(36), 0.4, 0.6, ["predict_{}".format(label_col)])
        df_ret = score_fit(df_ret, predict_origin.get(24), 0.4, 0.6, ["predict_{}".format(label_col)])
        df_ret = score_fit(df_ret, predict_origin.get(12), 0.4, 0.6, ["predict_{}".format(label_col)])
        return df_ret

    def result_modify(self, df_feats, df_result, submit_day):
        predict_begin = datetime.datetime.strptime(submit_day, "%Y-%m-%d") + datetime.timedelta(hours=24)
        range_start = str(predict_begin - datetime.timedelta(hours=24+self.delay_hour))[:19]
        range_end = str(predict_begin - datetime.timedelta(hours=1+self.delay_hour))[:19]
        time_range = "{}#{}".format(range_start, range_end)
        time_range_list = [time_range]
        df_feats_use = df_feats[df_feats['time_range'].isin(time_range_list)]
        df_feats_use = df_feats_use.reset_index(drop=True)

        df_sum = df_feats_use[['station_id']]
        for label in ["PM25", "PM10", "O3"]:
            model_file = "model_beijing/{}_BJ_XGB_PERIOD_24_LABEL_36.pkl".format(label)
            model_dict = cPickle.load(open(model_file, 'rb'))
            model_xgb = model_dict['estimator']
            feats_list = model_dict['feat_names']
            feat_use = xgb.DMatrix(df_feats_use.ix[:, feats_list].get_values())
            xgb_predict = model_xgb.predict(feat_use)
            df_sum[label] = np.maximum(xgb_predict, 0)

        if df_sum.shape[0] != 35:
            print "ERROR!"
            df_ret = None
        else:
            pm25_sum = dict()
            pm10_sum = dict()
            o3_sum = dict()
            for index in range(df_sum.shape[0]):
                line_dict = df_sum.loc[index, :].to_dict()
                pm25_sum[line_dict['station_id']] = line_dict["PM25"]
                pm10_sum[line_dict['station_id']] = line_dict["PM10"]
                o3_sum[line_dict['station_id']] = line_dict["O3"]

            df_ret_list = []
            trust_cnt = 36
            df_result['station_id'] = df_result['key_col'].apply(lambda x: x.split("#")[0])
            for station_id, df_s in df_result.groupby("station_id"):
                df_s = df_s.sort_values(by="key_col", ascending=True)
                df_doubt = df_s.head(trust_cnt)
                df_trust = df_s.tail(48 - trust_cnt)
                avg_other = (pm25_sum.get(station_id) - df_trust['predict_PM2.5'].sum()) / float(df_doubt.shape[0])
                avg_modify = avg_other - df_doubt['predict_PM2.5'].mean()
                mod_ratio = avg_modify / df_doubt['predict_PM2.5'].mean()
                mod_ratio = np.minimum(mod_ratio, BJ_MODIFY_TH)
                mod_ratio = np.maximum(mod_ratio, 0)
                avg_modify = df_doubt['predict_PM2.5'].mean() * mod_ratio
                df_doubt['predict_PM2.5'] = df_doubt['predict_PM2.5'].apply(lambda x: x + avg_modify)

                avg_other = (pm10_sum.get(station_id) - df_trust['predict_PM10'].sum()) / float(df_doubt.shape[0])
                avg_modify = avg_other - df_doubt['predict_PM10'].mean()
                mod_ratio = avg_modify / df_doubt['predict_PM10'].mean()
                mod_ratio = np.minimum(mod_ratio, BJ_MODIFY_TH)
                mod_ratio = np.maximum(mod_ratio, 0)
                avg_modify = df_doubt['predict_PM10'].mean() * mod_ratio
                df_doubt['predict_PM10'] = df_doubt['predict_PM10'].apply(lambda x: x + avg_modify)

                avg_other = (o3_sum.get(station_id) - df_trust['predict_O3'].sum()) / float(df_doubt.shape[0])
                avg_modify = avg_other - df_doubt['predict_O3'].mean()
                mod_ratio = avg_modify / df_doubt['predict_O3'].mean()
                mod_ratio = np.minimum(mod_ratio, BJ_MODIFY_TH)
                mod_ratio = np.maximum(mod_ratio, 0)
                avg_modify = df_doubt['predict_O3'].mean() * mod_ratio
                df_doubt['predict_O3'] = df_doubt['predict_O3'].apply(lambda x: x + avg_modify)

                df_ret_list.append(df_trust)
                df_ret_list.append(df_doubt)
            df_ret = pd.concat(df_ret_list, axis=0)
            df_ret = df_ret.drop('station_id', axis=1)
            df_ret = df_ret.sort_values(by="key_col", ascending=True)
            df_ret = df_ret.reset_index(drop=True)
            df_ret = df_ret.applymap(lambda x: 0.0 if isinstance(x, np.float) and x < 0.0 else x)
            print "修正后的df_ret: {}".format(df_ret.shape)
        return df_ret

    def aq_min_cal(self, df_result):
        df_use = df_result.copy()
        bj_aq_min = cPickle.load(open("data_beijing/utils_data/bj_station_aq_min.pkl", "r"))
        for item in ["PM2.5", "PM10", "O3"]:
            df_use['predict_{}'.format(item)] = df_use.apply(cal_aq_by_min, args=(item, bj_aq_min,), axis=1)
        return df_use

    def master(self):
        aq_item = ["predict_PM2.5", "predict_PM10", "predict_O3"]
        for date in get_every_day(self.submit_begin, self.submit_end):
            print "submit date: {}".format(date)
            # 特征提取: 时间范围 gap_max 48 + period 24
            current_time = datetime.datetime.now()
            predict_begin = datetime.datetime.strptime(date, "%Y-%m-%d") + datetime.timedelta(hours=24)
            time_list = []
            for i in range(1, 72):
                time_list.append(str(predict_begin - datetime.timedelta(hours=(72 - i + self.delay_hour)))[:19])
            feats_creator = FeatureCreate()
            df_feats, df_feats_s = feats_creator.master(time_list, date)
            use_time = (datetime.datetime.now() - current_time).total_seconds() / 60.0
            print "extract features use time: {:.2f} mints".format(use_time)

            # 调用无time gap的模型进行评分
            pm25_df = self.predict_no_gap(df_feats, hours=6, chose_label="PM25", submit_day=date)
            pm10_df = self.predict_no_gap(df_feats, hours=6, chose_label="PM10", submit_day=date)
            o3_df = self.predict_no_gap(df_feats, hours=6, chose_label="O3", submit_day=date)
            ret_no_gap = pm25_df.merge(pm10_df, on='key_col', how='outer')
            ret_no_gap = ret_no_gap.merge(o3_df, on='key_col', how='outer')
            print "ret_no_gap: {}".format(ret_no_gap.shape)

            # 调用有time gap的模型进行评分
            pm25_df = self.predict_with_gap(df_feats, chose_label="PM25", submit_day=date)
            pm10_df = self.predict_with_gap(df_feats, chose_label="PM10", submit_day=date)
            o3_df = self.predict_with_gap(df_feats, chose_label="O3", submit_day=date)
            ret_with_gap = pm25_df.merge(pm10_df, on='key_col', how='outer')
            ret_with_gap = ret_with_gap.merge(o3_df, on='key_col', how='outer')
            print "ret_with_gap: {}".format(ret_with_gap.shape)

            # 合并预测结果, 保存到最终文件
            result = score_fit(ret_with_gap, ret_no_gap, 0.5, 0.5, aq_item)
            result.to_csv(self.result_path + "{}.csv".format(date), index=False)

            # 修正预测结果
            if BJ_MODIFY:
                result_modify = self.result_modify(df_feats, result, date)
                if result_modify is not None:
                    ret_final = result_modify
                else:
                    ret_final = result
            else:
                ret_final = result

            # 校准预测值的下限
            if BJ_CAL:
                ret_final = self.aq_min_cal(ret_final)
            ret_final.to_csv(self.result_path + "{}.csv".format(date), index=False, encoding="utf-8")
        return


if __name__ == '__main__':
    predictor = PredictBeijingAqByLIHAO()
    predictor.submit_begin = SUBMIT_DATE
    predictor.submit_end = SUBMIT_DATE
    predictor.master()
