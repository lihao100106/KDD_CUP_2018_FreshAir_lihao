# coding:utf-8
import pandas as pd
import numpy as np
import cPickle
import operator
from math import radians, cos
from config import LD_FILL_NA
pd.options.mode.chained_assignment = None


class FeatureCreate(object):
    def __init__(self):
        self.path = "data_london/"
        self.predict_period = 24 * 1
        self.nearest_top_num = 5
        self.label_N = 1
        self.aq_item = ['PM25', 'PM10', 'NO2']
        self.meo_item = ['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed']
        self.time_list = []

    def basic_feature_aq(self, df):
        df_time = pd.DataFrame(self.time_list, columns=['utc_time'])
        feat_list = []
        for station_id, df_s in df.groupby('station_id'):
            df_s = df_time.merge(df_s, on='utc_time', how='left')
            df_s = df_s.sort_values(by='utc_time', ascending=True)
            df_s = df_s.reset_index(drop=True)
            utc_time_list = df_s['utc_time'].tolist()
            pm25_list = df_s['PM2.5'].tolist()
            pm10_list = df_s['PM10'].tolist()
            no2_list = df_s['NO2'].tolist()
            sample_id = 1
            i = 0
            while (i + self.predict_period) <= df_s.shape[0]:
                begin_time = utc_time_list[i]
                end_time = utc_time_list[i + self.predict_period - 1]
                feats_dict = dict()
                feats_dict['station_id'] = station_id.split('_')[0]
                feats_dict['sample_id'] = "{}_{}".format(feats_dict['station_id'], sample_id)
                feats_dict['time_range'] = "{}#{}".format(begin_time, end_time)
                for j in range(self.predict_period):
                    feats_dict['PM25_{}hours_ago'.format(self.predict_period - j)] = pm25_list[i + j]
                    feats_dict['PM10_{}hours_ago'.format(self.predict_period - j)] = pm10_list[i + j]
                    feats_dict['NO2_{}hours_ago'.format(self.predict_period - j)] = no2_list[i + j]
                feat_list.append(feats_dict)
                i += 1
                sample_id += 1
        feat_df = pd.DataFrame(feat_list)
        feat_df = feat_df.reset_index(drop=True)
        print "basic_feature_aq: {}".format(feat_df.shape)
        return feat_df

    def basic_feature_grid_meo(self, df):
        df_time = pd.DataFrame(self.time_list, columns=['utc_time'])
        feat_list = []
        for grid_id, df_s in df.groupby('stationName'):
            df_s = df_time.merge(df_s, on='utc_time', how='left')
            df_s = df_s.sort_values(by='utc_time', ascending=True)
            df_s = df_s.reset_index(drop=True)

            utc_time_list = df_s['utc_time'].tolist()
            temperature_list = df_s['temperature'].tolist()
            pressure_list = df_s['pressure'].tolist()
            humidity_list = df_s['humidity'].tolist()
            wind_d_list = df_s['wind_direction'].tolist()
            wind_s_list = df_s['wind_speed/kph'].tolist()
            i = 0
            while (i + self.predict_period) <= df_s.shape[0]:
                begin_time = utc_time_list[i]
                end_time = utc_time_list[i + self.predict_period - 1]
                feats_dict = dict()
                feats_dict['grid_id'] = grid_id
                feats_dict['time_range'] = "{}#{}".format(begin_time, end_time)
                for j in range(self.predict_period):
                    feats_dict['temperature_{}hours_ago'.format(self.predict_period - j)] = temperature_list[i + j]
                    feats_dict['pressure_{}hours_ago'.format(self.predict_period - j)] = pressure_list[i + j]
                    feats_dict['humidity_{}hours_ago'.format(self.predict_period - j)] = humidity_list[i + j]
                    feats_dict['wind_direction_{}hours_ago'.format(self.predict_period - j)] = wind_d_list[i + j]
                    feats_dict['wind_speed_{}hours_ago'.format(self.predict_period - j)] = wind_s_list[i + j]
                feat_list.append(feats_dict)
                i += 1
        feat_df = pd.DataFrame(feat_list)
        feat_df = feat_df.reset_index(drop=True)
        return feat_df

    def distance_feature_nearest(self, df_grid_all, feats_aq):
        distance_dict = cPickle.load(open(self.path + "utils_data/all_distance.pkl", "r"))
        direction_dict = cPickle.load(open(self.path + "utils_data/all_direction.pkl", "r"))
        distance_df_list = []
        for station_id, df_station in feats_aq.groupby('station_id'):
            distance_feats_df = df_station[['sample_id']]
            d_sort = sorted(distance_dict[station_id].items(), key=operator.itemgetter(1), reverse=False)
            # 选择距离station最近的5个grid的meo作为特征
            chose_grid_list = []
            chose_grid_cnt = 0
            for k, v in d_sort:
                if k.startswith('london_grid') and chose_grid_cnt < self.nearest_top_num:
                    chose_grid_list.append(k)
                    chose_grid_cnt += 1
            df_info = df_station[['sample_id', 'station_id', 'time_range']]

            for nearest_n in range(len(chose_grid_list)):
                grid_id = chose_grid_list[nearest_n]
                nearest_n += 1
                df_grid = df_grid_all[df_grid_all['grid_id'] == grid_id]
                df_grid = df_grid.reset_index(drop=True)
                df_info["distance_{}".format(grid_id)] = df_info['station_id'].apply(lambda x: 1 / distance_dict[x][grid_id])
                df_info["direction_{}".format(grid_id)] = df_info['station_id'].apply(lambda x: direction_dict[x][grid_id])
                # 注意: direction_dict[x][grid_id] 是 grid_id -> station_id的方向, 对 distance_dict 取了倒数(效果和距离成反比)
                feats_tmp = df_info.merge(df_grid, on='time_range', how='left')
                feats_name_list = ['sample_id']
                for i in range(self.predict_period):
                    col_1 = 'temperature_{}hours_ago'.format(i + 1)
                    col_2 = 'pressure_{}hours_ago'.format(i + 1)
                    col_3 = 'humidity_{}hours_ago'.format(i + 1)
                    col_d = 'wind_direction_{}hours_ago'.format(i + 1)
                    col_s = 'wind_speed_{}hours_ago'.format(i + 1)
                    feats_tmp['direction_angle'] = feats_tmp[col_d] - feats_tmp["direction_{}".format(grid_id)]
                    feats_tmp['angle_cosine'] = feats_tmp['direction_angle'].apply(lambda x: cos(radians(x)))
                    # 温度/气压/湿度：使用方向和风速进行修正
                    for col in [col_1, col_2, col_3]:
                        col_name = "{}_modify".format(col)
                        col_list = [col, col_s, 'angle_cosine', "distance_{}".format(grid_id)]
                        feats_tmp[col_name] = feats_tmp[col_list[0]] * feats_tmp[col_list[1]] * feats_tmp[col_list[2]] * feats_tmp[col_list[3]]
                        feats_name_list.append(col_name)

                    # 风速：使用方向进行修正
                    for col in [col_s]:
                        col_name = "{}_modify".format(col_s)
                        col_list = [col, 'angle_cosine', "distance_{}".format(grid_id)]
                        feats_tmp[col_name] = feats_tmp[col_list[0]] * feats_tmp[col_list[1]] * feats_tmp[col_list[2]]
                        feats_name_list.append(col_name)

                    # 风向：使用原始数据
                    for col in [col_d]:
                        col_name = "{}_origin".format(col)
                        feats_tmp[col_name] = feats_tmp[col]
                        feats_name_list.append(col_name)
                feats_tmp_use = feats_tmp[feats_name_list]
                feats_tmp_use = feats_tmp_use.rename(columns=lambda x: "nearest_NO{}_grid#{}".format(nearest_n, x))
                feats_tmp_use = feats_tmp_use.rename(columns={"nearest_NO{}_grid#sample_id".format(nearest_n): "sample_id"})
                distance_feats_df = distance_feats_df.merge(feats_tmp_use, on='sample_id', how='inner')
            distance_df_list.append(distance_feats_df)
        ret_df = pd.concat(distance_df_list, axis=0)
        ret_df = ret_df.reset_index(drop=True)
        print "distance_feature_nearest: {}".format(ret_df.shape)
        return ret_df

    def distance_nearest_station(self, top_num, feats_aq, df_station_other):
        distance_dict = cPickle.load(open(self.path + "utils_data/all_distance.pkl", "r"))

        station_aq_dict = dict()
        for station_id, df_station in df_station_other.groupby('station_id'):
            station_aq_dict[station_id] = df_station.drop('sample_id', axis=1)
        for station_id, df_station in feats_aq.groupby('station_id'):
            station_aq_dict[station_id] = df_station.drop('sample_id', axis=1)

        distance_df_list = []
        for station_id, df_station in feats_aq.groupby('station_id'):
            distance_feats_df = df_station[['sample_id']]
            d_sort = sorted(distance_dict[station_id].items(), key=operator.itemgetter(1), reverse=False)
            # 选择距离station最近的3个station的station作为特征
            chose_grid_list = []
            chose_grid_cnt = 0
            for k, v in d_sort:
                if not k.startswith('london_grid') and chose_grid_cnt < top_num:
                    chose_grid_list.append(k)
                    chose_grid_cnt += 1
            df_info = df_station[['sample_id', 'station_id', 'time_range']]

            for nearest_n in range(len(chose_grid_list)):
                station_id = chose_grid_list[nearest_n]
                df_station_id = station_aq_dict.get(station_id)
                df_info["distance_{}".format(station_id)] = df_info['station_id'].apply(
                    lambda x: 50 - distance_dict[x][station_id])
                # 注意: direction_dict[x][grid_id] 是 grid_id -> station_id的方向, 对 distance_dict 取了倒数(效果和距离成反比)
                feats_tmp = df_info.merge(df_station_id, on='time_range', how='left')
                feats_name_list = ['sample_id']
                for i in range(self.predict_period):
                    col_1 = 'PM25_{}hours_ago'.format(i + 1)
                    col_2 = 'PM10_{}hours_ago'.format(i + 1)
                    col_3 = 'NO2_{}hours_ago'.format(i + 1)
                    # 使用距离进行修正
                    for col in [col_1, col_2, col_3]:
                        col_name = "{}_modify".format(col)
                        col_list = [col, "distance_{}".format(station_id)]
                        feats_tmp[col_name] = feats_tmp[col_list[0]] * feats_tmp[col_list[1]]
                        feats_name_list.append(col_name)
                feats_tmp_use = feats_tmp[feats_name_list]
                feats_tmp_use = feats_tmp_use.rename(
                    columns=lambda x: "nearest_NO{}_station#{}".format(nearest_n + 1, x))
                feats_tmp_use = feats_tmp_use.rename(
                    columns={"nearest_NO{}_station#sample_id".format(nearest_n + 1): "sample_id"})
                distance_feats_df = distance_feats_df.merge(feats_tmp_use, on='sample_id', how='inner')
            distance_df_list.append(distance_feats_df)
        ret_df = pd.concat(distance_df_list, axis=0)
        ret_df = ret_df.reset_index(drop=True)
        print "distance_nearest_station: {}".format(ret_df.shape)
        return ret_df

    def design_feature_aq(self, feats_aq):
        # 过去N个小时的 meo 均值,方差,最大值,最小值,总和,增长率,增长率的变化率,增长率的均值,增长率的方差,残差
        feats_list = []
        for index in range(feats_aq.shape[0]):
            design_feat = dict()
            row = feats_aq.loc[index, :].to_dict()
            design_feat['sample_id'] = row['sample_id']
            for aq in self.aq_item:
                aq_data_list = []
                for hour in range(self.predict_period):
                    key_name = "{}_{}hours_ago".format(aq, hour + 1)
                    aq_data_list.append(row[key_name])

                # 计算aq_data_list的基本统计特征
                for n in range(2, self.predict_period + 1):
                    aq_np = np.array(aq_data_list[:n])
                    new_key_pre = "{}_last_{}hours_".format(aq, n)
                    design_feat[new_key_pre + "sum"] = aq_np.sum()
                    design_feat[new_key_pre + "max"] = aq_np.max()
                    design_feat[new_key_pre + "min"] = aq_np.min()
                    design_feat[new_key_pre + "avg"] = aq_np.mean()
                    design_feat[new_key_pre + "std"] = aq_np.std()

                # 计算aq_data_list的增长率和残差
                aq_rate_list = []
                aq_res_list = []
                for n in range(len(aq_data_list) - 1):
                    new = aq_data_list[n]
                    old = aq_data_list[n + 1]
                    increase_res = new - old
                    if old > 0.0:
                        increase_rate = increase_res / float(old)
                    else:
                        increase_rate = np.nan
                    aq_rate_list.append(increase_rate)
                    aq_res_list.append(increase_res)
                    new_key_rate = "{}_last_{}hours_increase_rate".format(aq, n)
                    new_key_res = "{}_last_{}hours_residual".format(aq, n)
                    design_feat[new_key_rate] = increase_rate
                    design_feat[new_key_res] = increase_res

                # 增长率的统计特征
                for n in range(2, len(aq_rate_list) + 1):
                    aq_rate_np = np.array(aq_rate_list[:n])
                    new_key_pre = "{}_last_{}hours_increase_rate_".format(aq, n)
                    design_feat[new_key_pre + "sum"] = aq_rate_np.sum()
                    design_feat[new_key_pre + "max"] = aq_rate_np.max()
                    design_feat[new_key_pre + "min"] = aq_rate_np.min()
                    design_feat[new_key_pre + "avg"] = aq_rate_np.mean()
                    design_feat[new_key_pre + "std"] = aq_rate_np.std()

                # 残差的统计特征
                for n in range(2, len(aq_res_list) + 1):
                    aq_res_np = np.array(aq_res_list[:n])
                    new_key_pre = "{}_last_{}hours_residual_".format(aq, n)
                    design_feat[new_key_pre + "sum"] = aq_res_np.sum()
                    design_feat[new_key_pre + "max"] = aq_res_np.max()
                    design_feat[new_key_pre + "min"] = aq_res_np.min()
                    design_feat[new_key_pre + "avg"] = aq_res_np.mean()
                    design_feat[new_key_pre + "std"] = aq_res_np.std()
            feats_list.append(design_feat)
        feats_df = pd.DataFrame(feats_list)
        feats_df = feats_df.reset_index(drop=True)
        print "design_feature_aq: {}".format(feats_df.shape)
        return feats_df

    def design_feature_grid(self, top_n, df_grid):
        # 过去N个小时的 meo 均值,方差,最大值,最小值,总和,增长率,增长率的变化率,增长率的均值,增长率的方差,残差
        feats_list = []
        for index in range(df_grid.shape[0]):
            design_feat = dict()
            row = df_grid.loc[index, :].to_dict()
            design_feat['sample_id'] = row['sample_id']
            for meo in self.meo_item:
                if meo in ['wind_direction']:
                    end_with = 'origin'
                else:
                    end_with = 'modify'
                meo_data_list = []
                for hour in range(self.predict_period):
                    key_name = "nearest_NO{}_grid#{}_{}hours_ago_{}".format(top_n, meo, hour + 1, end_with)
                    meo_data_list.append(row[key_name])

                # 计算aq_data_list的基本统计特征
                for n in range(2, self.predict_period + 1):
                    meo_np = np.array(meo_data_list[:n])
                    new_key_pre = "nearest_NO{}_grid#{}_last_{}hours_".format(top_n, meo, n)
                    design_feat[new_key_pre + "sum"] = meo_np.sum()
                    design_feat[new_key_pre + "max"] = meo_np.max()
                    design_feat[new_key_pre + "min"] = meo_np.min()
                    design_feat[new_key_pre + "avg"] = meo_np.mean()
                    design_feat[new_key_pre + "std"] = meo_np.std()

                # 计算meo_data_list的增长率和残差
                meo_rate_list = []
                meo_res_list = []
                for n in range(len(meo_data_list) - 1):
                    new = meo_data_list[n]
                    old = meo_data_list[n + 1]
                    increase_res = new - old
                    increase_rate = increase_res / float(old)
                    meo_rate_list.append(increase_rate)
                    meo_res_list.append(increase_res)
                    new_key_rate = "nearest_NO{}_grid#{}_last_{}hours_increase_rate".format(top_n, meo, n)
                    new_key_res = "nearest_NO{}_grid#{}_last_{}hours_residual".format(top_n, meo, n)
                    design_feat[new_key_rate] = increase_rate
                    design_feat[new_key_res] = increase_res

                # 增长率的统计特征
                for n in range(2, len(meo_rate_list) + 1):
                    meo_rate_np = np.array(meo_rate_list[:n])
                    new_key_pre = "nearest_NO{}_grid#{}_last_{}hours_increase_rate_".format(top_n, meo, n)
                    design_feat[new_key_pre + "sum"] = meo_rate_np.sum()
                    design_feat[new_key_pre + "max"] = meo_rate_np.max()
                    design_feat[new_key_pre + "min"] = meo_rate_np.min()
                    design_feat[new_key_pre + "avg"] = meo_rate_np.mean()
                    design_feat[new_key_pre + "std"] = meo_rate_np.std()

                # 残差的统计特征
                for n in range(2, len(meo_res_list) + 1):
                    meo_res_np = np.array(meo_res_list[:n])
                    new_key_pre = "nearest_NO{}_grid#{}_last_{}hours_residual_".format(top_n, meo, n)
                    design_feat[new_key_pre + "sum"] = meo_res_np.sum()
                    design_feat[new_key_pre + "max"] = meo_res_np.max()
                    design_feat[new_key_pre + "min"] = meo_res_np.min()
                    design_feat[new_key_pre + "avg"] = meo_res_np.mean()
                    design_feat[new_key_pre + "std"] = meo_res_np.std()

            feats_list.append(design_feat)
        feats_df = pd.DataFrame(feats_list)
        feats_df = feats_df.reset_index(drop=True)
        print "design_feature_grid: {}".format(feats_df.shape)
        return feats_df

    def design_feature_aq_nearest_station(self, top_n, df_aq):
        # 过去N个小时的 aq 均值,方差,最大值,最小值,总和,增长率,增长率的变化率,增长率的均值,增长率的方差,残差
        feats_list = []
        for index in range(df_aq.shape[0]):
            design_feat = dict()
            row = df_aq.loc[index, :].to_dict()
            design_feat['sample_id'] = row['sample_id']
            for item in self.aq_item:
                end_with = 'modify'
                meo_data_list = []
                for hour in range(self.predict_period):
                    key_name = "nearest_NO{}_station#{}_{}hours_ago_{}".format(top_n, item, hour + 1, end_with)
                    meo_data_list.append(row[key_name])

                # 计算aq_data_list的基本统计特征
                for n in range(2, self.predict_period + 1):
                    meo_np = np.array(meo_data_list[:n])
                    new_key_pre = "nearest_NO{}_station#{}_last_{}hours_".format(top_n, item, n)
                    design_feat[new_key_pre + "sum"] = meo_np.sum()
                    design_feat[new_key_pre + "max"] = meo_np.max()
                    design_feat[new_key_pre + "min"] = meo_np.min()
                    design_feat[new_key_pre + "avg"] = meo_np.mean()
                    design_feat[new_key_pre + "std"] = meo_np.std()

                # 计算meo_data_list的增长率和残差
                meo_rate_list = []
                meo_res_list = []
                for n in range(len(meo_data_list) - 1):
                    new = meo_data_list[n]
                    old = meo_data_list[n + 1]
                    increase_res = new - old
                    if old > 0.0:
                        increase_rate = increase_res / float(old)
                    else:
                        increase_rate = np.nan
                    meo_rate_list.append(increase_rate)
                    meo_res_list.append(increase_res)
                    new_key_rate = "nearest_NO{}_station#{}_last_{}hours_increase_rate".format(top_n, item, n)
                    new_key_res = "nearest_NO{}_station#{}_last_{}hours_residual".format(top_n, item, n)
                    design_feat[new_key_rate] = increase_rate
                    design_feat[new_key_res] = increase_res

                # 增长率的统计特征
                for n in range(2, len(meo_rate_list) + 1):
                    meo_rate_np = np.array(meo_rate_list[:n])
                    new_key_pre = "nearest_NO{}_station#{}_last_{}hours_increase_rate_".format(top_n, item, n)
                    design_feat[new_key_pre + "sum"] = meo_rate_np.sum()
                    design_feat[new_key_pre + "max"] = meo_rate_np.max()
                    design_feat[new_key_pre + "min"] = meo_rate_np.min()
                    design_feat[new_key_pre + "avg"] = meo_rate_np.mean()
                    design_feat[new_key_pre + "std"] = meo_rate_np.std()

                # 残差的统计特征
                for n in range(2, len(meo_res_list) + 1):
                    meo_res_np = np.array(meo_res_list[:n])
                    new_key_pre = "nearest_NO{}_station#{}_last_{}hours_residual_".format(top_n, item, n)
                    design_feat[new_key_pre + "sum"] = meo_res_np.sum()
                    design_feat[new_key_pre + "max"] = meo_res_np.max()
                    design_feat[new_key_pre + "min"] = meo_res_np.min()
                    design_feat[new_key_pre + "avg"] = meo_res_np.mean()
                    design_feat[new_key_pre + "std"] = meo_res_np.std()

            feats_list.append(design_feat)
        feats_df = pd.DataFrame(feats_list)
        feats_df = feats_df.reset_index(drop=True)
        print "design_feature_aq_nearest_station: {}".format(feats_df.shape)
        return feats_df

    def master(self, time_range, submit_date):
        self.time_list = time_range
        if LD_FILL_NA:
            df_aq_fore = pd.read_table(self.path + "use_data/{}_aq_forecast_filled.csv".format(submit_date), sep=",")
        else:
            df_aq_fore = pd.read_table(self.path + "use_data/{}_aq_forecast.csv".format(submit_date), sep=",")
        df_aq_other = pd.read_table(self.path + "use_data/{}_aq_other.csv".format(submit_date), sep=",")
        df_grid_meo = pd.read_table(self.path + "use_data/{}_grid_meo.csv".format(submit_date), sep=",")

        aq_feat_df_fore = self.basic_feature_aq(df_aq_fore)
        aq_feat_df_other = self.basic_feature_aq(df_aq_other)
        meo_feat_df = self.basic_feature_grid_meo(df_grid_meo)
        nearest_grid_meo_feat_df = self.distance_feature_nearest(meo_feat_df, aq_feat_df_fore)
        nearest_station_feat_df = self.distance_nearest_station(3, aq_feat_df_fore, aq_feat_df_other)
        aq_design_feat_df_fore = self.design_feature_aq(aq_feat_df_fore)
        nearest_grid_meo_design_feat_df = self.design_feature_grid(top_n=1, df_grid=nearest_grid_meo_feat_df)
        nearest_station_design_feat_df = self.design_feature_aq_nearest_station(1, nearest_station_feat_df)

        feats_df_list = [
            aq_feat_df_fore,
            nearest_grid_meo_feat_df,
            nearest_station_feat_df,
            nearest_grid_meo_design_feat_df,
            aq_design_feat_df_fore,
            nearest_station_design_feat_df
        ]
        ret_df = feats_df_list[0]
        for i in range(1, len(feats_df_list)):
            ret_df = ret_df.merge(feats_df_list[i], on='sample_id', how="inner")
        print "特征df维度: {}".format(ret_df.shape)
        return ret_df
