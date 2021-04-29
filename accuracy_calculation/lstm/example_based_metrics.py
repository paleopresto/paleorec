# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 11:50:34 2021

@author: shrav
"""

import pandas as pd
import numpy as np
import sys
import os
import json
from sys import platform as _platform

if _platform == "win32":
    sys.path.insert(1, '..\..\prediction\lstm\\')
else:
    sys.path.insert(1, '../../prediction/lstm//')    

# from LSTMpredict import LSTMpredict
from LSTMpredict_copy import LSTMpredict

if _platform == "win32":
    sys.path.insert(1, '..\..\\')
else:
    sys.path.insert(1, '../../')
from utils import fileutils

if _platform == "win32":
    predict_obj = LSTMpredict(model_file_path='..\..\data\model_lstm\\', ground_truth_file_path='..\..\data\ground_truth_info\\', topk=5)
    test_data_path = '..\..\data\csv\\'
    ground_truth_label_path = '..\..\data\ground_truth_info\\'
else:
    predict_obj = LSTMpredict(model_file_path='../../data/model_lstm/', ground_truth_file_path='../../data/ground_truth_info/', topk=5)
    test_data_path = '../../data/csv/'
    ground_truth_label_path = '../../data/ground_truth_info/'

df_test = pd.read_csv(fileutils.get_latest_file_with_path(test_data_path, 'lipdverse_test_*.csv'))
# df_test = pd.read_csv('D:\\annotating_paleoclimate_data\\paleorec\\data\\csv\\lipdverse_test_1234.csv')
df_test = df_test.replace(np.nan, 'NA', regex=True)
df_test = df_test.replace(',', '', regex=True)
df_test_list = df_test.values.tolist()
len_dict = predict_obj.len_dict

inverse_ref_dict = {v:k for k,v in predict_obj.reference_dict.items()}

ground_truth_label_file = fileutils.get_latest_file_with_path(ground_truth_label_path, 'ground_truth_label_*.json')
with open(ground_truth_label_file, 'r') as json_file:
    ground_truth = json.load(json_file)
    
ground_truth_label_dict = ground_truth['ground_truth']
accuracy_list = []
precision_list = []
recall_list = []
good_list = []
total_acc, total_pres, total_rec = 0,0,0
avg_acc = {"1": [], "2": [], "3": [], "3_units": [], "4": [], "5": [], "6": []}
avg_pres = {"1": [], "2": [], "3": [], "3_units": [], "4": [], "5": [], "6": []}
avg_rec = {"1": [], "2": [], "3": [], "3_units": [], "4": [], "5": [], "6": []}

# Yi = ground truth set for input
# hxi = predicted set for the input
# xi = input label for which next word is predicted

def get_acc_score(xi, hxi):
    '''
    Calculate Accuracy Score for LSTM prediction. 
    The function returns 10 if the actual value from input matches the 1st string in the list of top 5 predictions using LSTM.
    Else it returns 5 if the actual value is present in the list of top 5 predictions using LSTM.
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
    if xi in ground_truth_label_dict:
        Yi = set(ground_truth_label_dict[xi])
        
        if hxi[0] == 'NA' or hxi[0] == 'NotApplicable':
            hxi = hxi[:1]
        
        if len(Yi) < len(hxi):
            hxi = hxi[:len(Yi)] 
        
        hxi = set(hxi)
        
        acc = len(Yi.intersection(hxi))/len(Yi.union(hxi))
        return acc
    return 0

def get_precision_score(xi, hxi):

    if xi in ground_truth_label_dict:
        Yi = set(ground_truth_label_dict[xi])
        
        if hxi[0] == 'NA' or hxi[0] == 'NotApplicable':
            hxi = hxi[:1]

        if len(Yi) < len(hxi):
            hxi = hxi[:len(Yi)]

        hxi = set(hxi)

        precision = len(Yi.intersection(hxi))/len(hxi)
        return precision
    return 0


def get_recall_score(xi, hxi):

    if xi in ground_truth_label_dict:
        Yi = set(ground_truth_label_dict[xi])
        
        if hxi[0] == 'NA' or hxi[0] == 'NotApplicable':
            hxi = hxi[:1]

        if len(Yi) < len(hxi):
            hxi = hxi[:len(Yi)]
            
        hxi = set(hxi)

        recall = len(Yi.intersection(hxi))/len(Yi)
        return recall
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
    
    Depending on previous accuracy calculations we have received accuracy score for LSTM predictions = 7.68571
    If the average prediction for a sentence crosses this mark, we can consider LSTM to be a good fit for predictions for this archiveType.
    
    
    Returns
    -------
    None.

    '''
    global accuracy_list, precision_list, recall_list, df_test_list
    
    
    c = 0
    for lis in df_test_list:
        lis = [val.replace(" ", "") for val in lis]
        acc, pres, rec = 0, 0, 0
        
        i = 1
        
        while i < 7:
            
            ref_word = inverse_ref_dict[lis[i-1]] if lis[i-1] in inverse_ref_dict else ''
            
            actual_len = len(ground_truth_label_dict[ref_word]) if ref_word in ground_truth_label_dict else -1
            if i < 2:
                input_sent_list = lis[:i]
                results = predict_obj.predictForSentence(sentence=','.join(input_sent_list))['0']
                received_len = len(results)
                if not results:
                    results = ['NA']
                acc += get_acc_score(lis[i-1], results)
                pres += get_precision_score(lis[i-1], results)
                rec += get_recall_score(lis[i-1], results)
            elif i == 2:
                input_sent_list = lis[:i]
                results_untis =  predict_obj.predictForSentence(sentence=','.join(input_sent_list))['0']
                results = predict_obj.predictForSentence(sentence=','.join(input_sent_list))['1']
                
                received_len = len(results)

                if not results_untis:
                    results_untis = ['NA']
                if not results:
                    results = ['NA']
                acc += get_acc_score(lis[i-1], results_untis)
                pres += get_precision_score(lis[i-1], results_untis)
                rec += get_recall_score(lis[i-1], results_untis)

                acc += get_acc_score(lis[i-1], results)
                pres += get_precision_score(lis[i-1], results)
                rec += get_recall_score(lis[i-1], results)
                
                i += 1
            else:
                temp = lis[:2]
                temp.extend(lis[3:i])
                input_sent_list = temp
                results = predict_obj.predictForSentence(sentence=','.join(input_sent_list))['0']
                received_len = len(results)
                if not results:
                    results = ['NA']
                acc += get_acc_score(lis[i-1], results)
                pres += get_precision_score(lis[i-1], results)
                rec += get_recall_score(lis[i-1], results)
            i += 1             
            c += 1
            if c < 30:
                print('actual length = {}, received length = {}'.format(actual_len, received_len))
        accuracy_list.append(acc/7)
        precision_list.append(pres/7)
        recall_list.append(rec/7)
        
        
    print('Accuracy for LSTM = ', sum(accuracy_list)/len(accuracy_list))
    print('Precision for LSTM = ', sum(precision_list)/len(precision_list))
    print('Recall for LSTM = ', sum(recall_list)/len(recall_list))

def calculate_score_for_test_data_chain():

    global accuracy_list, precision_list, recall_list, df_test_list
    global total_acc, total_pres, total_rec

    df = pd.DataFrame(np.nan, index = [0,1,2,3],columns=['archiveType', 'proxyObsType', 'units', 'interpretation/variable', 'interpretation/variableDetail', 'inferredVariable', 'inferredVarUnits', 'accuracy_score', 'precision_score', 'recall_score'])
    
    c = 0
    chain_count = 1
    for lis in df_test_list:
        lis = [val.replace(" ", "") for val in lis]
        i = 1
        
        while i < 7:
            
            ref_word = inverse_ref_dict[lis[i-1]] if lis[i-1] in inverse_ref_dict else ''
            if c < 12:
                print('ref word {} for lis item {}'.format(ref_word, lis[i-1]))
            actual_len = len(ground_truth_label_dict[ref_word]) if ref_word in ground_truth_label_dict else -1

            if i < 2:
                input_sent_list = lis[:i]
                results = predict_obj.predictForSentence(sentence=','.join(input_sent_list))['0']
                received_len = len(results)
                if not results:
                    results = ['NA']
                # acc = get_acc_score(lis[i-1], results)/len_dict['2']
                # pres = get_precision_score(lis[i-1], results)/len_dict['2']
                # rec = get_recall_score(lis[i-1], results)/len_dict['2']

                acc = get_acc_score(lis[i-1], results)
                pres = get_precision_score(lis[i-1], results)
                rec = get_recall_score(lis[i-1], results)

                total_acc += acc
                total_pres += pres
                total_rec += rec
                
                values_to_add = {'archiveType': lis[i-1], 'proxyObsType': lis[i], 'accuracy_score' : acc, 'precision_score' : pres, 'recall_score' : rec}
                row_to_add = pd.Series(values_to_add, name = chain_count)
                df = df.append(row_to_add)
                chain_count += 1

            elif i == 2:
                input_sent_list = lis[:i]
                results_untis =  predict_obj.predictForSentence(sentence=','.join(input_sent_list))['0']
                results = predict_obj.predictForSentence(sentence=','.join(input_sent_list))['1']
                
                received_len = len(results)

                if not results_untis:
                    results_untis = ['NA']
                if not results:
                    results = ['NA']

                acc = get_acc_score(lis[i-1], results_untis)/len_dict['3_units']
                pres = get_precision_score(lis[i-1], results_untis)/len_dict['3_units']
                rec = get_recall_score(lis[i-1], results_untis)/len_dict['3_units']

                # acc = get_acc_score(lis[i-1], results_untis)
                # pres = get_precision_score(lis[i-1], results_untis)
                # rec = get_recall_score(lis[i-1], results_untis)

                total_acc += acc
                total_pres += pres
                total_rec += rec

                values_to_add = {'archiveType': lis[i-2], 'proxyObsType': lis[i-1], 'units': lis[i], 'accuracy_score' : acc, 'precision_score' : pres, 'recall_score' : rec}
                row_to_add = pd.Series(values_to_add, name = chain_count)
                df = df.append(row_to_add)
                chain_count += 1
                
                acc = get_acc_score(lis[i-1], results)/len_dict['3']
                pres = get_precision_score(lis[i-1], results)/len_dict['3']
                rec = get_recall_score(lis[i-1], results)/len_dict['3']

                # acc = get_acc_score(lis[i-1], results)
                # pres = get_precision_score(lis[i-1], results)
                # rec = get_recall_score(lis[i-1], results)

                total_acc += acc
                total_pres += pres
                total_rec += rec

                values_to_add = {'archiveType': lis[i-2], 'proxyObsType': lis[i-1],'interpretation/variable' : lis[i+1], 'accuracy_score' : acc, 'precision_score' : pres, 'recall_score' : rec}
                row_to_add = pd.Series(values_to_add, name = chain_count)
                df = df.append(row_to_add)
                chain_count += 1

                i += 1
            else:
                temp = lis[:2]
                temp.extend(lis[3:i])
                input_sent_list = temp
                results = predict_obj.predictForSentence(sentence=','.join(input_sent_list))['0']
                received_len = len(results)
                if not results:
                    results = ['NA']
                acc = get_acc_score(lis[i-1], results)/len_dict[str(i)]
                pres = get_precision_score(lis[i-1], results)/len_dict[str(i)]
                rec = get_recall_score(lis[i-1], results)/len_dict[str(i)]

                # acc = get_acc_score(lis[i-1], results)
                # pres = get_precision_score(lis[i-1], results)
                # rec = get_recall_score(lis[i-1], results)

                total_acc += acc
                total_pres += pres
                total_rec += rec

                values_to_add = {'archiveType': lis[0], 'proxyObsType': lis[1], 'interpretation/variable': lis[3], 'accuracy_score' : acc, 'precision_score' : pres, 'recall_score' : rec}
                if i == 4:
                    values_to_add['interpretation/variableDetail'] = lis[i]
                elif i == 5:
                    values_to_add['interpretation/variableDetail'] = lis[4]
                    values_to_add['inferredVariable'] = lis[i]
                elif i == 6:
                    values_to_add['interpretation/variableDetail'] = lis[4]
                    values_to_add['inferredVariable'] = lis[5]
                    values_to_add['inferredVarUnits'] = lis[i]

                row_to_add = pd.Series(values_to_add, name = chain_count)
                df = df.append(row_to_add)
                chain_count += 1

            i += 1             
            c += 1
            if c < 13:
                print('actual length = {}, received length = {}'.format(actual_len, received_len))        

    total_acc = total_acc/chain_count
    total_pres = total_pres/chain_count
    total_rec = total_rec/chain_count

    print('Accuracy for chains for LSTM = ', total_acc)
    print('Precision for chains for LSTM = ', total_pres)
    print('Recall for chains for LSTM = ', total_rec)

    accuracy_data_path = os.path.join(test_data_path, 'accuracy_prediction_lstm_chains.csv')
    df = df.replace(np.nan, '', regex=True)
    df.to_csv(accuracy_data_path, sep = ',', encoding = 'utf-8',index = False)

def store_results_to_csv():
    '''
    Append the accuracy score for each row.
    Also append the information which signifies whether LSTM is a fit for predictions for this archiveType.
    Store this information back to a csv file.

    Returns
    -------
    None.

    '''
    global df_test
    
    accuracy_data_path = os.path.join(test_data_path, 'accuracy_prediction_lstm.csv')
    df_test = df_test.assign(accuracy_score=pd.Series(accuracy_list).values)
    df_test = df_test.assign(precision_score=pd.Series(precision_list).values)
    df_test = df_test.assign(recall_score=pd.Series(recall_list).values)
    df_test.to_csv(accuracy_data_path, sep = ',', encoding = 'utf-8',index = False)
    
if __name__ == "__main__":
    # calculate_score_for_test_data()
    print('**************************************************************************************')
    calculate_score_for_test_data_chain()
    # store_results_to_csv()
else:
    calculate_score_for_test_data()
    store_results_to_csv()