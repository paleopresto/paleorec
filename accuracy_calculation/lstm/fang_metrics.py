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
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


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
df_test = df_test.replace(np.nan, 'NA', regex=True)
df_test = df_test.replace(',', '', regex=True)
df_test_list = df_test.values.tolist()
len_dict = predict_obj.len_dict

inverse_ref_dict = {v:k for k,v in predict_obj.reference_dict.items()}

ground_truth_label_file = fileutils.get_latest_file_with_path(ground_truth_label_path, 'ground_truth_label_*.json')
with open(ground_truth_label_file, 'r') as json_file:
    ground_truth = json.load(json_file)
    
ground_truth_label_dict = ground_truth['ground_truth']

avg_rec_3 = {"2": [], "3": [], "3_units": [], "4": [], "5": [], "6": []}
avg_rec_5 = {"2": [], "3": [], "3_units": [], "4": [], "5": [], "6": []}
avg_rec_7 = {"2": [], "3": [], "3_units": [], "4": [], "5": [], "6": []}
avg_rec_10 = {"2": [], "3": [], "3_units": [], "4": [], "5": [], "6": []}
avg_rec_12 = {"2": [], "3": [], "3_units": [], "4": [], "5": [], "6": []}
avg_rec_14 = {"2": [], "3": [], "3_units": [], "4": [], "5": [], "6": []}
avg_rec_16 = {"2": [], "3": [], "3_units": [], "4": [], "5": [], "6": []}

avg_mrr_3 = {"2": [], "3": [], "3_units": [], "4": [], "5": [], "6": []}
avg_mrr_5 = {"2": [], "3": [], "3_units": [], "4": [], "5": [], "6": []}
avg_mrr_7 = {"2": [], "3": [], "3_units": [], "4": [], "5": [], "6": []}
avg_mrr_10 = {"2": [], "3": [], "3_units": [], "4": [], "5": [], "6": []}

# Yi = ground truth set for input
# hxi = predicted set for the input
# xi = input label for which next word is predicted
# grnd_truth_i = ground truth value from chain

def get_mrr_score(grnd_truth_i,hxi):
    
    '''
    Formula from https://arxiv.org/pdf/1808.06414.pdf 'Next Item Recommendation with Self-Attention' pg 6
    '''

    if hxi[0] == 'NA' or hxi[0] == 'NotApplicable':
            hxi = hxi[:1]

    # if grnd_truth_i == hxi[0]:
    #     return 1
    # elif grnd_truth_i in hxi:
    #     return 1/(hxi.index(grnd_truth_i)+1)
    # else:
    #     return 0

    res = []
    for lim in [3,5,7,10]:
        temp_hxi = hxi[:lim]
        if grnd_truth_i == temp_hxi[0]:
            res.append(1)
        elif grnd_truth_i in temp_hxi:
            res.append(1/(temp_hxi.index(grnd_truth_i)+1))
        else:
            res.append(0)
    return res

def get_recall_score(grnd_truth_i, hxi):

    '''
    Formula from https://arxiv.org/pdf/1808.06414.pdf 'Next Item Recommendation with Self-Attention' pg 5 
    '''
    if hxi[0] == 'NA' or hxi[0] == 'NotApplicable':
            hxi = hxi[:1]

    r3 = 1 if grnd_truth_i in hxi[:3] else 0
    r5 = 1 if grnd_truth_i in hxi[:5] else 0
    r7 = 1 if grnd_truth_i in hxi[:7] else 0
    r10 = 1 if grnd_truth_i in hxi[:10] else 0
    r12 = 1 if grnd_truth_i in hxi[:12] else 0
    r14 = 1 if grnd_truth_i in hxi[:14] else 0
    r16 = 1 if grnd_truth_i in hxi[:16] else 0

    # if grnd_truth_i in hxi:
    #     return 1
    # else:
    #     return 0

    return r3, r5, r7, r10, r12, r14, r16

def mean_recall(val):
    '''
    Method to calculate the average recall for all the examples in the test data for a particular chain length across all the recommendation set size.(3, 5, 7, 10 ,12, 14, 16)

    Parameters
    ----------
    val : int
        Chain length across which mean is calculated.

    Returns
    -------
    res : list
        List containing the average recall across a particular chain size for different recommendation set sizes.

    '''
    objs_list = [avg_rec_3[val], avg_rec_5[val], avg_rec_7[val], avg_rec_10[val], avg_rec_12[val], avg_rec_14[val], avg_rec_16[val] ]
    res = []
    for obj in objs_list:
        res.append(sum(obj)/len(obj))
    return res

def mean_mrr(val):
    '''
    Method to calculate the MRR for a chain sizes across different recommendation set sizes.(3, 5, 7, 10)

    Parameters
    ----------
    val : int
        Chain length across which mean is calculated

    Returns
    -------
    res : list
        list containing the average MRR across a particular chain length for different recommendation set sizes.

    '''
    
    objs_list = [avg_mrr_3[val], avg_mrr_5[val], avg_mrr_7[val], avg_mrr_10[val]]
    res = []
    for obj in objs_list:
        res.append(sum(obj)/len(obj))
    return res

def calculate_score_for_test_data_chain():
    '''
    Calculates the evaluation metrics for the provided test data by generating different length chains from the input.
    Generates a line chart of the average recall and MRR across different recommednation set sizes.

    Returns
    -------
    None.

    '''

    global accuracy_list, precision_list, recall_list, df_test_list
    global total_acc, total_pres, total_rec

    df = pd.DataFrame(np.nan, index = [0,1,2,3],columns=['archiveType', 'proxyObsType', 'units', 'interpretation/variable', 'interpretation/variableDetail', 'inferredVariable', 'inferredVarUnits', 'accuracy_score', 'precision_score', 'recall_score'])
    
    chain_count = 1
    for lis in df_test_list:
        lis = [val.replace(" ", "") for val in lis]
        i = 1
        
        while i < 7:
            
            if i < 2:
                input_sent_list = lis[:i]
                results = predict_obj.predictForSentence(sentence=','.join(input_sent_list))['0']
                received_len = len(results)
                if not results:
                    results = ['NA']

                r3, r5, r7, r10, r12, r14, r16 = get_recall_score(lis[i], results)
                m3, m5, m7, m10 = get_mrr_score(lis[i], results)

                # avg_mrr['2'].append(get_mrr_score(lis[i], results))
                avg_mrr_3['2'].append(m3)
                avg_mrr_5['2'].append(m5)
                avg_mrr_7['2'].append(m7)
                avg_mrr_10['2'].append(m10)

                # avg_rec['2'].append(get_recall_score(lis[i], results))
                avg_rec_3['2'].append(r3)
                avg_rec_5['2'].append(r5)
                avg_rec_7['2'].append(r7)
                avg_rec_10['2'].append(r10)
                avg_rec_12['2'].append(r12)
                avg_rec_14['2'].append(r14)
                avg_rec_16['2'].append(r16)

                values_to_add = {'archiveType': lis[i-1], 'proxyObsType': lis[i], 'Recall' : avg_rec_10['2'][-1], 'MRR' : avg_mrr_10['2'][-1]}
                row_to_add = pd.Series(values_to_add, name = chain_count)
                df = df.append(row_to_add)
                chain_count += 1

            elif i == 2:
                input_sent_list = lis[:i]
                results_units =  predict_obj.predictForSentence(sentence=','.join(input_sent_list))['0']
                results = predict_obj.predictForSentence(sentence=','.join(input_sent_list))['1']
                
                received_len = len(results)

                if not results_units:
                    results_units = ['NA']
                if not results:
                    results = ['NA']

                r3, r5, r7, r10, r12, r14, r16 = get_recall_score(lis[i], results_units)
                m3, m5, m7, m10 = get_mrr_score(lis[i], results_units)

                # avg_mrr['3_units'].append(get_mrr_score(lis[i], results_units))
                avg_mrr_3['3_units'].append(m3)
                avg_mrr_5['3_units'].append(m5)
                avg_mrr_7['3_units'].append(m7)
                avg_mrr_10['3_units'].append(m10)

                # avg_rec['3_units'].append(get_recall_score(lis[i], results_units))
                avg_rec_3['3_units'].append(r3)
                avg_rec_5['3_units'].append(r5)
                avg_rec_7['3_units'].append(r7)
                avg_rec_10['3_units'].append(r10)
                avg_rec_12['3_units'].append(r12)
                avg_rec_14['3_units'].append(r14)
                avg_rec_16['3_units'].append(r16)


                values_to_add = {'archiveType': lis[i-2], 'proxyObsType': lis[i-1], 'units': lis[i], 'Recall' : avg_rec_10['3_units'][-1], 'MRR' : avg_mrr_10['3_units'][-1]}
                row_to_add = pd.Series(values_to_add, name = chain_count)
                df = df.append(row_to_add)
                chain_count += 1

                r3, r5, r7, r10, r12, r14, r16 = get_recall_score(lis[i+1], results)
                m3, m5, m7, m10 = get_mrr_score(lis[i], results)

                # avg_mrr['3'].append(get_mrr_score(lis[i+1], results))
                avg_mrr_3['3'].append(m3)
                avg_mrr_5['3'].append(m5)
                avg_mrr_7['3'].append(m7)
                avg_mrr_10['3'].append(m10)
                # avg_rec['3'].append(get_recall_score(lis[i+1], results))
                avg_rec_3['3'].append(r3)
                avg_rec_5['3'].append(r5)
                avg_rec_7['3'].append(r7)
                avg_rec_10['3'].append(r10)
                avg_rec_12['3'].append(r12)
                avg_rec_14['3'].append(r14)
                avg_rec_16['3'].append(r16)

                values_to_add = {'archiveType': lis[i-2], 'proxyObsType': lis[i-1],'interpretation/variable' : lis[i+1], 'Recall' : avg_rec_10['3'][-1], 'MRR' : avg_mrr_10['3'][-1]}
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

                r3, r5, r7, r10, r12, r14, r16 = get_recall_score(lis[i], results)
                m3, m5, m7, m10 = get_mrr_score(lis[i], results)

                # avg_mrr[str(i)].append(get_mrr_score(lis[i], results))
                avg_mrr_3[str(i)].append(m3)
                avg_mrr_5[str(i)].append(m5)
                avg_mrr_7[str(i)].append(m7)
                avg_mrr_10[str(i)].append(m10)
                # avg_rec[str(i)].append(get_recall_score(lis[i], results))
                avg_rec_3[str(i)].append(r3)
                avg_rec_5[str(i)].append(r5)
                avg_rec_7[str(i)].append(r7)
                avg_rec_10[str(i)].append(r10)
                avg_rec_12[str(i)].append(r12)
                avg_rec_14[str(i)].append(r14)
                avg_rec_16[str(i)].append(r16)

                values_to_add = {'archiveType': lis[0], 'proxyObsType': lis[1], 'interpretation/variable': lis[3], 'Recall' : avg_rec_10[str(i)][-1], 'MRR' : avg_mrr_10[str(i)][-1]}
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

    print('**************************************************************************************')

    accuracy_data_path = os.path.join(test_data_path, 'accuracy_prediction_fang_metrics_1.csv')
    df = df.replace(np.nan, '', regex=True)
    df.to_csv(accuracy_data_path, sep = ',', encoding = 'utf-8',index = False)


    # plotting the data
    plt.figure(num=1, figsize=(8, 6), dpi=80)
    # create data
    x = [3,5,7,10,12,14,16]
    y2 = mean_recall('2')
    y3 = mean_recall('3')
    y3_u = mean_recall('3_units')
    y4 = mean_recall('4')
    y5 = mean_recall('5')
    y6 = mean_recall('6')
    
    # plot lines
    plt.plot(x, y2, '-o', label = "Chain till Proxy(len = 2)")
    plt.plot(x, y3, '-o', label = "Chain till Interp Var(len = 3)")
    plt.plot(x, y3_u, '-o', label = "Chain till Proxy Units(len = 3)")
    plt.plot(x, y4, '-o', label = "Chain till Interp Var Det(len = 4)")
    plt.plot(x, y5, '-o', label = "Chain till Inf Var(len = 5)")
    plt.plot(x, y6, '-o', label = "Chain till Inf Var Units(len = 6)")
    plt.xlabel('Recommendation Set Sizes(3,5,7,10,12,14,16)', fontsize = 12)
    plt.title('Recall/Hit Ratio for Recommendation Set Size(3,5,7,10,12,14,16)', fontsize=14)
    plt.ylim(0,1.1)
    plt.legend()
    plt.show()

    print('*************************************************************************************************')

    plt.figure(num=2, figsize=(8, 6), dpi=80)
    x = [3,5,7,10]
    y2 = mean_mrr('2')
    y3 = mean_mrr('3')
    y3_u = mean_mrr('3_units')
    y4 = mean_mrr('4')
    y5 = mean_mrr('5')
    y6 = mean_mrr('6')

    # plot lines
    plt.plot(x, y2, '-o', label = "Chain till Proxy(len = 2)")
    plt.plot(x, y3, '-o', label = "Chain till Interp Var(len = 3)")
    plt.plot(x, y3_u, '-o', label = "Chain till Proxy Units(len = 3)")
    plt.plot(x, y4, '-o', label = "Chain till Interp Var Det(len = 4)")
    plt.plot(x, y5, '-o', label = "Chain till Inf Var(len = 5)")
    plt.plot(x, y6, '-o', label = "Chain till Inf Var Units(len = 6)")
    plt.xlabel('Recommendation Set Sizes(3,5,7,10)', fontsize = 12)
    plt.title('MRR for Recommendation Set Size(3,5,7,10)', fontsize=14)
    plt.ylim(0,1.1)
    plt.legend()
    plt.show()



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
    
    accuracy_data_path = os.path.join(test_data_path, 'accuracy_prediction_fang_metrics.csv')
    df_test = df_test.assign(accuracy_score=pd.Series(accuracy_list).values)
    df_test = df_test.assign(precision_score=pd.Series(precision_list).values)
    df_test = df_test.assign(recall_score=pd.Series(recall_list).values)
    df_test.to_csv(accuracy_data_path, sep = ',', encoding = 'utf-8',index = False)
    
if __name__ == "__main__":
    calculate_score_for_test_data_chain()