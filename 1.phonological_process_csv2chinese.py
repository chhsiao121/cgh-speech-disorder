# 將統計好的標記結果由原本的音律歷程標號改成中文的csv檔
from ast import literal_eval
import pandas as pd
import os
import shutil
import json

WORDCARD = "data_0820word"
DATE = "0820_0916"
STATPATH = '/D/TWCC/work/cgh_2022/jsonmv/stat/' + \
    WORDCARD+'/'+DATE+'/compareST_'+DATE+'.csv'
SAVECSVPATH = '/D/TWCC/work/cgh_2022/jsonmv/stat/data_0820word/0820_0916/forsat.csv'

class_dict = {
    "塞音化": 1,
    "母音化": 2,
    "母音省略": 3,
    "舌前音化": 4,
    "舌根音化": 5,
    "不送氣音化": 6,
    "聲隨韻母": 7,
    "邊音化": 8,
    "齒間音": 9,
    "子音省略": 10,
    "擦音化": 11,
    "介音省略": 12,
    "塞擦音化": 13,
    "複韻母省略": 14,
    "其他": 15,
    "正確": 16,
    "雜訊無法辨識": 17
}

df_stat = pd.read_csv(STATPATH, index_col=0)

for case in df_stat.columns.tolist():
    for wordcard in df_stat.index.tolist():
        if not df_stat[case][wordcard]:
            pass
        elif (df_stat[case][wordcard] != df_stat[case][wordcard]):
            pass
        else:
            df_stat[case][wordcard] = literal_eval(
                df_stat[case][wordcard])  # convert to list

for case in df_stat.columns.tolist():
    for wordcard in df_stat.index.tolist():
        if not df_stat[case][wordcard]:
            pass
        listtmp = []
        for aaa in df_stat[case][wordcard]:
            tmp = list(class_dict.keys())[list(class_dict.values()).index(aaa)]
            listtmp.append(tmp)
        df_stat[case][wordcard] = listtmp
df_stat.to_csv(SAVECSVPATH)
