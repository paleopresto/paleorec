# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 17:21:35 2021

@author: shrav
"""

import json
from heapq import heappop, heappush, heapify 

# read model file
with open('model.txt', 'r') as f:
  model = json.load(f)

archives_map = model['archives_map']
q0_chain1 = model['q0_chain1']
q0_chain2 = model['q0_chain2']
transition_matrix_chain1 = model['transition_matrix_chain1']
transition_matrix_chain2 = model['transition_matrix_chain2']

  
  
def back_track(data, transition_prob_dict):
    '''
    Function to get top 5 items for each item in sequence

    Parameters
    ----------
    data : list/str
        Input sequence.
    transition_prob_dict : dict
        To calculate log probability of the sequence.

    Returns
    -------
    list
        Output list for the input sequence.

    '''
    outlist = []
    if type(data) is list:
        for content in data:
            outlist.append(back_track(content))
    else:
        prob, word = data
        if word not in transition_prob_dict:
            return []
        pq = []
        heapify(pq)
        for key, val in transition_prob_dict[word].items():
            if len(pq) < 5:
                heappush(pq, ((prob+val), key))
            else:
                if (prob+val) > pq[0][0]:
                    blah = heappop(pq)
                    heappush(pq, ((prob+val), key))
        
        temp = []
        while pq:
            tup = heappop(pq)
            temp.append(tup)
        return temp[::-1]
    
    return outlist


def predict_seq(sentence, initial_prob_dict, chain_length, transition_prob_dict):
    '''
    Predict the top 5 elements at each stage for every item in the chain
    There are 2 chain types:
        archive -> proxyObservationType -> units, 
        archive -> proxyObservationType -> interpretation/variable, interpretation/variableDetail

    Parameters
    ----------
    sentence : str
        Input sequence.
    initial_prob_dict : dict
        Store initial probabilities corresponding to the chain type.
    chain_length : int
        To generate predictions corresponding to chain length.
    transition_prob_dict : dict
        Stores transitiion probabilites for item to every other item. 

    Returns
    -------
    output_list : list
        List in hierarchial fashion containing top 5 predictions for value at each level.
    
    example 
    input: 'Wood'
    output: [[(-1.0286697494934511, 'Wood')], 
             [[(-2.8598709507728035, 'Trsgi'), (-3.519116579657067, 'ARS'), (-3.588109451144019, 'EPS'), (-3.701438136451022, 'SD'), (-3.701438136451022, 'Core')]], 
             [[
               [(-3.5698252496491296, 'NA'), (-7.647362693554849, 'Wood'), (-7.647362693554849, 'MXD'), (-7.647362693554849, 'LakeSediment'), (-7.647362693554849, 'Composite')], 
               [(-4.628778704511761, 'NA'), (-8.029976086173917, 'Wood'), (-8.029976086173917, 'MXD'), (-8.029976086173917, 'LakeSediment'), (-8.029976086173917, 'Composite')], 
               [(-4.744541310700955, 'NA'), (-8.076745820876159, 'Wood'), (-8.076745820876159, 'MXD'), (-8.076745820876159, 'LakeSediment'), (-8.076745820876159, 'Composite')], 
               [(-4.936909607836329, 'NA'), (-8.15578543270453, 'Wood'), (-8.15578543270453, 'MXD'), (-8.15578543270453, 'LakeSediment'), (-8.15578543270453, 'Composite')], 
               [(-4.971198681314961, 'NA'), (-6.803780145063271, 'NotApplicable'), (-8.190074506183162, 'Wood'), (-8.190074506183162, 'MXD'), (-8.190074506183162, 'Composite')]
             ]]
            ]
        

    '''
    sentence = sentence.strip().split(',')
    output_list = []
    for index, word in enumerate(sentence):
        #initial probability
        prob = 0
        if index == 0:
            if word in archives_map:
                word = archives_map[word]
            prob = initial_prob_dict[word]
            output_list.append([(prob, word)])
        else:
            if word in transition_prob_dict[sentence[index-1]]:
                prob += transition_prob_dict[sentence[index-1]][word]
                output_list.append([(prob, word)])
            elif 'NA' in transition_prob_dict[sentence[index-1]]:
                prob += transition_prob_dict[sentence[index-1]]['NA']
                output_list.append([(prob, word)])
            # else:
            # call API to add new word to the data
    
    while len(output_list) < chain_length:
        output_list.append(back_track(output_list[-1]), transition_prob_dict)
    
    return output_list



# ********************** run locally ******************************
# initital probability dict according to the chain
# initial_prob_dict = q0_chain1
# # initial_prob_dict = q0_chain2

# # chain length = archive -> proxyObservationType -> units (3)
# # chain length = archive -> proxyObservationType -> interpretation/variable -> interpretation/variableDetail (4)
# chain_length = 3
# # chain_length = 4

# # transition probability dict according to the chain
# transition_prob_dict = transition_matrix_chain1
# # transition_prob_dict = transition_matrix_chain2

# test_data = open ('test_archive_1.txt', 'r', encoding='utf-8')    
# output_file = open ('output_archive_proxy_units.txt', 'w', encoding='utf-8')
# for line in test_data:
#     output_list = predict_seq(line, initial_prob_dict, chain_length, transition_prob_dict)
#     output_file.write(str(output_list) + '\n')

  