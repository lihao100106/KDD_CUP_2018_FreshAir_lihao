# coding=utf-8
from config import *
from get_data_from_API import data_update_master
from final_model_beijing import PredictBeijingAqByLIHAO
from final_model_london import PredictLondonAqByLIHAO
from utils import lh_result_merge


def check_params():
    print "\n当前的参数配置如下:"
    print "\n选择进行预测的城市: {}".format(CITY_LIST)
    if BJ_FILL_NA:
        print "北京：aq缺失值使用预测值填充"
    else:
        print "北京：缺失值不进行填充"

    if BJ_MODIFY:
        print "北京：使用修正模型"
    else:
        print "北京：不使用修正模型"

    if BJ_CAL:
        print "北京：对预测结果的最小值 进行校准"
    else:
        print "北京：对预测结果的最小值不进行校准"

    if LD_FILL_NA:
        print "伦敦：aq缺失值使用预测值填充"
    else:
        print "伦敦：缺失值不进行填充"

    if LD_MODIFY:
        print "伦敦：使用修正模型"
    else:
        print "伦敦：不使用修正模型"

    if LD_CAL:
        print "伦敦：对预测结果的最小值 进行校准"
    else:
        print "伦敦：对预测结果的最小值不进行校准"

    if IS_TEST:
        print "当前是测试模式: 对{} 23:00:00的数据手动剔除\n".format(SUBMIT_DATE)
    else:
        print "当前是线上模式: 使用能够获取到的所有最新数据预测\n"
    return


def lh_kdd_master(city_list, date):
    print "\n提交日期: {}".format(date)
    begin_time = datetime.datetime.now()
    print "\n开始获取{}的数据...".format(date)
    data_update_master(city_list, date)

    if 'bj' in city_list:
        print "\n开始预测北京:"
        bj_predictor = PredictBeijingAqByLIHAO()
        bj_predictor.submit_begin = date
        bj_predictor.submit_end = date
        bj_predictor.master()

    if 'ld' in city_list:
        print "\n开始预测伦敦:"
        ld_predictor = PredictLondonAqByLIHAO()
        ld_predictor.submit_begin = date
        ld_predictor.submit_end = date
        ld_predictor.master()

    print "\n合并预测结果.."
    lh_result_merge(date)

    end_time = datetime.datetime.now()
    print "\n本次预测总共耗时{:.2f}分钟\n".format((end_time - begin_time).total_seconds() / 60.0)
    return


if __name__ == '__main__':
    check_params()
    lh_kdd_master(CITY_LIST, SUBMIT_DATE)
