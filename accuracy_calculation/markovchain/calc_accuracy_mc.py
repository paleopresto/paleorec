# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 13:19:52 2021

@author: shrav
"""

import pandas as pd
import numpy as np
import sys
import os
from sys import platform as _platform

if _platform == "win32":
    sys.path.insert(1, '..\..\prediction\markovchain\\')
else:
    sys.path.insert(1, '../../prediction/markovchain//')
from MCpredict import MCpredict

if _platform == "win32":
    sys.path.insert(1, '..\..\\')
else:
    sys.path.insert(1, '../../')
from utils import fileutils

if _platform == "win32":
    test_data_path = '..\..\data\csv\\'
else:
    test_data_path = '../../data/csv/'

df_test = pd.read_csv(fileutils.get_latest_file_with_path(test_data_path, 'lipdverse_test_*.csv'))
df_test = df_test.replace(np.nan, 'NA', regex=True)
df_test_list = df_test.values.tolist()


accuracy_list = []
good_list = []

if _platform == "win32":
    model_file_path='..\..\data\model_mc\\'
else:
    model_file_path='../../data/model_lstm/'

pred3 = MCpredict(3, 5, model_file_path)
pred4 = MCpredict(4, 5, model_file_path)

def getScoreForResult(test_val, result_list):
    '''
    Calculate Accuracy Score for Markov Chain prediction. 
    The function returns 10 if the actual value from input matches the 1st string in the list of top 5 predictions using Markov Chains.
    Else it returns 5 if the actual value is present in the list of top 5 predictions using Markov Chains.
    Else it returns 0.
    

    Parameters
    ----------
    test_val : string
        Actual value for the test input.
    result_list : list
        List consisting of the predictions using Markov Chains.

    Returns
    -------
    int
        Accuracy score depending on where the actual value is present in list of predicted values.

    '''
    if result_list:
        if test_val == result_list[0]:
            return 10
        elif test_val in result_list:
            return 5
        else:
            return 0
    return 0

def calculate_score_for_test_data():
    '''
    This method will generate the list of top 5 predictions for each sentence combination in the test input.
    Each row in the test input consists of 7 fields; 
    archiveType, proxyObservationType, units, interpretation/variable, interpretation/variableDetail, inferredVariable, inferredVarUnits.
    Since we have 2 chains for prediction, we will split the sentence accordingly.
    1st prediction will be to get the proxyObservationType given the archiveType as a comma-separated sentence.
    2nd prediction will be to get the units and interpretation/variable given the archiveType and proxyObservationType as a comma-separated sentence.
    3rd prediction will be to get the interpretation/variableDetail given archiveType, proxyObservationType, interpretation/variable as a comma-separated sentence
    and so on...
    
    For each sentence that is created, get the accuracy score using the actual value in test input and the list of predictions.
    
    Calculate an average score of predictions for each combination of input sentence.
    
    Depending on previous accuracy calculations we have received accuracy score for Markov Chain predictions = 7.17143
    If the average prediction for a sentence crosses this mark, we can consider Markov Chain to be a good fit for predictions for this archiveType.
    
    
    Returns
    -------
    None.

    '''
    global accuracy_list, good_list, df_test_list
    
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
        good_list.append(True if acc/7 >= 7.17143 else False)
    
    print('Accuracy for Markov Chain = ', sum(accuracy_list)/len(accuracy_list))
            
def store_results_to_csv():
    '''
    Append the accuracy score for each row.
    Also append the information which signifies whether Markov Chain is a fit for predictions for this archiveType.
    Store this information back to a csv file.

    Returns
    -------
    None.

    '''
    global df_test
    
    accuracy_data_path = os.path.join(test_data_path, 'accuracy_prediction_lstm.csv')
    df_test = df_test.assign(accuracy_score=pd.Series(accuracy_list).values)
    df_test = df_test.assign(isMCGood=pd.Series(good_list).values)
    df_test.to_csv(accuracy_data_path, sep = ',', encoding = 'utf-8',index = False)

if __name__ == "__main__":
    calculate_score_for_test_data()
    store_results_to_csv()
else:
    calculate_score_for_test_data()
    store_results_to_csv()


        