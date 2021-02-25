# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 13:19:52 2021

@author: shrav
"""

import pandas as pd
import numpy as np
import sys
import os

# FOR WINDOWS
# sys.path.insert(1, '..\..\prediction\markovchain\\')
# FOR LINUX
sys.path.insert(1, '../../prediction/markovchain//')
from MCpredict import MCpredict

# FOR WINDOWS
# sys.path.insert(1, '..\..\\')
# FOR LINUX
sys.path.insert(1, '../../')
from utils import fileutils

# FOR WINDOWS
# test_data_path = '..\..\data\csv\\'
# FOR LINUX
test_data_path = '../../data/csv/'

df_test = pd.read_csv(fileutils.get_latest_file_with_path(test_data_path, 'lipdverse_test_*.csv'))
df_test = df_test.replace(np.nan, 'NA', regex=True)
df_test_list = df_test.values.tolist()

# print(df_test_list)
accuracy_list = []
good_list = []

# FOR WINDOWS
# model_file_path='..\..\data\model_mc\\'
# FOR LINUX
model_file_path='../../data/model_lstm/'

pred3 = MCpredict(3, 5, model_file_path)
pred4 = MCpredict(4, 5, model_file_path)

def getScoreForResult(test_val, result_list):
    if test_val == result_list[0]:
        return 10
    elif test_val in result_list:
        return 5
    else:
        return 0

for lis in df_test_list:
    acc = 0
    i = 1
    
    while i < 7:
        
        if i < 2:
            sentence = (',').join(lis[:i])
            result_list = pred4.predict_seq(sentence)['0']
            acc += getScoreForResult(lis[i], result_list)
        elif i == 2:
            sentence = (',').join(lis[:i])
            result_list_units = pred3.predict_seq(sentence)['0']
            result_list = pred4.predict_seq(sentence)['0']
            acc += getScoreForResult(lis[i], result_list_units)
            acc += getScoreForResult(lis[i+1], result_list)
            
            i += 1
        else:
            temp = lis[:2]
            temp.extend(lis[3:i])
            sentence = (',').join(temp)
            result_list = pred4.predict_seq(sentence)['0']
            acc += getScoreForResult(lis[i], result_list)
        i += 1
    accuracy_list.append(acc/7)
    good_list.append(True if acc/7 >= 7.0857 else False)

print('Accuracy for Markov Chain = ', sum(accuracy_list)/len(accuracy_list))
            
accuracy_data_path = os.path.join(test_data_path, 'accuracy_prediction_lstm.csv')
df_test = df_test.assign(accuracy_score=pd.Series(accuracy_list).values)
df_test = df_test.assign(isMCGood=pd.Series(good_list).values)
df_test.to_csv(accuracy_data_path, sep = ',', encoding = 'utf-8',index = False)
        