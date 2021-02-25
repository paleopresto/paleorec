# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 20:08:34 2021

@author: shrav
"""

import pandas as pd
import numpy as np
import sys
import os

# FOR WINDOWS
# sys.path.insert(1, '..\..\prediction\lstm\\')
# FOR LINUX
sys.path.insert(1, '../../prediction/lstm//')
from LSTMpredict import LSTMpredict

# FOR WINDOWS
# sys.path.insert(1, '..\..\\')
# FOR LINUX
sys.path.insert(1, '../../')
from utils import fileutils

# FOR WINDOWS
# predict_obj = LSTMpredict(model_file_path='..\..\data\model_lstm\\', mc_model_file_path='..\..\data\model_mc\\', topk=5)
# FOR LINUX
predict_obj = LSTMpredict(model_file_path='../../data/model_lstm/', mc_model_file_path='../../data/model_mc/', topk=5)

# FOR WINDOWS
# test_data_path = '..\..\data\csv\\'
# FOR LINUX
test_data_path = '../../data/csv/'

df_test = pd.read_csv(fileutils.get_latest_file_with_path(test_data_path, 'lipdverse_test_*.csv'))
df_test = df_test.replace(np.nan, 'NA', regex=True)
df_test_list = df_test.values.tolist()

def getScoreForResult(test_val, result_list):
    if test_val == result_list[0]:
        return 10
    elif test_val in result_list:
        return 5
    else:
        return 0
    

accuracy_list = []
good_list = []
for lis in df_test_list:
    lis = [val.replace(" ", "") for val in lis]
    acc = 0
    i = 1
    
    while i < 7:
        
        if i < 2:
            input_sent_list = lis[:i]
            results = predict_obj.predictForSentence(sentence=','.join(input_sent_list))['0']
            acc += getScoreForResult(lis[i], results)
        elif i == 2:
            input_sent_list = lis[:i]
            results_untis =  predict_obj.predictForSentence(sentence=','.join(input_sent_list))['0']
            results = predict_obj.predictForSentence(sentence=','.join(input_sent_list))['1']
            acc += getScoreForResult(lis[i], results_untis)
            acc += getScoreForResult(lis[i+1], results)
            
            i += 1
        else:
            temp = lis[:2]
            temp.extend(lis[3:i])
            input_sent_list = temp
            results = predict_obj.predictForSentence(sentence=','.join(input_sent_list))['0']
            acc += getScoreForResult(lis[i], results)
        i += 1
    accuracy_list.append(acc/7)
    good_list.append(True if acc/7 >= 7.68571 else False)
    
print('Accuracy for LSTM = ', sum(accuracy_list)/len(accuracy_list))


accuracy_data_path = os.path.join(test_data_path, 'accuracy_prediction_lstm.csv')
df_test = df_test.assign(accuracy_score=pd.Series(accuracy_list).values)
df_test = df_test.assign(isLSTMGood=pd.Series(good_list).values)
df_test.to_csv(accuracy_data_path, sep = ',', encoding = 'utf-8',index = False)