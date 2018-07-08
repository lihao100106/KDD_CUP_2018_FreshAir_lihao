# coding=utf-8
from config import *
import requests


def submit_file(path=None, file_name=None):
    if file_name:
        result_file = file_name
        if file_name.count("lihao_result_for_submit_") > 0:
            describe_txt = "lh+aqi merge"
        else:
            name_list = file_name.split(".csv")[0].split("_")
            describe_txt = " ".join(name_list[2:]) + " merge"
    else:
        result_file = RESULT_NAME
        describe_txt = "lihao origin version"
    if path:
        file_path = path
    else:
        file_path = RESULT_PATH

    print "submit file path: {}".format(file_path)
    print "submit file name: {}".format(result_file)
    print "file description: {}".format(describe_txt)

    files = {'files': open(file_path + result_file, 'rb')}
    data = {
        "user_id": "lihao100106",
        "team_token": "2f619dc81b2648a595b6b6c805729dd38ffc38d2f7440a84a9133f712bbdbf4a",
        "description": describe_txt,
        "filename": result_file,
    }
    url = 'https://biendata.com/competition/kdd_2018_submit/'
    response = requests.post(url, files=files, data=data, proxies=ce_proxies)
    print response.text
    return response.text


if __name__ == '__main__':
    submit_file(file_name="lihao_result_for_submit_{}.csv".format(SUBMIT_DATE))
    submit_file(path="data_result/score_merge/result_merge/",
                file_name="{}_merge_lh+sx+jq+zp+aqi.csv".format(SUBMIT_DATE))
