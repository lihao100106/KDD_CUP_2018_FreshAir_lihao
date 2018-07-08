# coding:utf-8
import datetime

CITY_LIST = ['bj', 'ld']
IS_TEST = False
BJ_FILL_NA = False
LD_FILL_NA = False
BJ_MODIFY = True
LD_MODIFY = True
BJ_CAL = True
LD_CAL = True

submit_time = datetime.datetime.now() - datetime.timedelta(hours=8)
# SUBMIT_DATE = str(submit_time)[:10]
SUBMIT_DATE = "2018-05-01"
RESULT_PATH = "../../output/result_lh/"
RESULT_NAME = "{}_lh.csv".format(SUBMIT_DATE)

BJ_MODIFY_TH = 0.1
LD_MODIFY_TH = 0.1
