# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 17:21:35 2021

@author: shrav
"""

import json
from heapq import heappop, heappush, heapify 

class MCpredict:

    def __init__(self, chain_length):
        '''
        Constructor to define object of Predict class.
        Thus we will have to read the model only once instead of having to read it every time we call the predict function.

        Parameters
        ----------
        chain_length : int
            Determine the type of prediction.
            There are 2 chain types:
            archive -> proxyObservationType -> units, 
            archive -> proxyObservationType -> interpretation/variable, interpretation/variableDetail

        Returns
        -------
        None.

        '''
        # read model file
        with open('model.txt', 'r') as f:
          model = json.load(f)
         
        self.archives_map = model['archives_map']
      
        if chain_length == 3:
            self.initial_prob_dict = model['q0_chain1']
            self.chain_length = chain_length
            self.transition_prob_dict = model['transition_matrix_chain1']
        else:
            self.initial_prob_dict = model['q0_chain2']
            self.chain_length = chain_length
            self.transition_prob_dict = model['transition_matrix_chain2']
         
    def get_inner_list(self, in_list):
        '''
        Backtracking code to recursively obtain the item name from the hierachial output list.

        Parameters
        ----------
        in_list : list/ tuple
            Either a list object or tuple whose data is retreived.
            

        Returns
        -------
        list
            Condensed hierarchial version of the list without probabilities.

        '''
        outlist = []
        if type(in_list) is list:
            for content in in_list:
                outlist.append(self.get_inner_list(content))
        else:
            return in_list[1]
        return outlist
         
    def pretty_output(self, output_list):
        '''
        Get the item list without the probabilities.

        Parameters
        ----------
        output_list : list
            Output List after complete processing..

        Returns
        -------
        out_dict : dict
            Ordered Dict with level as the key and value as the condensed list for each level.
            
        Example:
            input: [[(-1.0286697494934511, 'Wood')], [(-1.8312012012793524, 'Trsgi')], 
                    [[(-2.5411555001556785, 'NA'), (-6.618692944061398, 'Wood'), (-6.618692944061398, 'MXD'), (-6.618692944061398, 'LakeSediment'), (-6.618692944061398, 'Composite')]]]
            output: {'0': ['Wood'], '1': ['Trsgi'], '2': ['NA', 'Wood', 'MXD', 'LakeSediment', 'Composite']}


        '''
        
        out_dict = {}
        for i, l in enumerate(output_list):
            inner_list = self.get_inner_list(l)
            if type(inner_list[0]) == list:
                out_dict[str(i)] = inner_list[0]
            else:
                out_dict[str(i)] = inner_list
        
        return out_dict
      
    def back_track(self, data):
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
                outlist.append(self.back_track(content))
        else:
            prob, word = data
            if word not in self.transition_prob_dict:
                return []
            pq = []
            heapify(pq)
            for key, val in self.transition_prob_dict[word].items():
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
    
    
    def predict_seq(self, sentence):
        '''
        Predict the top 5 elements at each stage for every item in the chain
        There are 2 chain types:
            archive -> proxyObservationType -> units, 
            archive -> proxyObservationType -> interpretation/variable, interpretation/variableDetail
    
        Parameters
        ----------
        sentence : str
            Input sequence.
    
        Returns
        -------
        output_list : dict
            Dict in hierarchial fashion containing top 5 predictions for value at each level.
        
        example 
        input: 'Wood'
        intermediate output: [[(-1.0286697494934511, 'Wood')], 
                 [[(-2.8598709507728035, 'Trsgi'), (-3.519116579657067, 'ARS'), (-3.588109451144019, 'EPS'), (-3.701438136451022, 'SD'), (-3.701438136451022, 'Core')]], 
                 [[
                   [(-3.5698252496491296, 'NA'), (-7.647362693554849, 'Wood'), (-7.647362693554849, 'MXD'), (-7.647362693554849, 'LakeSediment'), (-7.647362693554849, 'Composite')], 
                   [(-4.628778704511761, 'NA'), (-8.029976086173917, 'Wood'), (-8.029976086173917, 'MXD'), (-8.029976086173917, 'LakeSediment'), (-8.029976086173917, 'Composite')], 
                   [(-4.744541310700955, 'NA'), (-8.076745820876159, 'Wood'), (-8.076745820876159, 'MXD'), (-8.076745820876159, 'LakeSediment'), (-8.076745820876159, 'Composite')], 
                   [(-4.936909607836329, 'NA'), (-8.15578543270453, 'Wood'), (-8.15578543270453, 'MXD'), (-8.15578543270453, 'LakeSediment'), (-8.15578543270453, 'Composite')], 
                   [(-4.971198681314961, 'NA'), (-6.803780145063271, 'NotApplicable'), (-8.190074506183162, 'Wood'), (-8.190074506183162, 'MXD'), (-8.190074506183162, 'Composite')]
                 ]]
                ]
        final output: {'0': ['Wood'], 
                       '1': ['Trsgi', 'ARS', 'EPS', 'SD', 'Core'], 
                       '2': [
                               ['NA', 'Wood', 'MXD', 'LakeSediment', 'Composite'], 
                               ['NA', 'Wood', 'MXD', 'LakeSediment', 'Composite'], 
                               ['NA', 'Wood', 'MXD', 'LakeSediment', 'Composite'], 
                               ['NA', 'Wood', 'MXD', 'LakeSediment', 'Composite'], 
                               ['NA', 'NotApplicable', 'Wood', 'MXD', 'Composite']
                            ]
                       }
    
        '''
        sentence = sentence.strip().split(',')
        output_list = []
        for index, word in enumerate(sentence):
            #initial probability
            prob = 0
            if index == 0:
                if word in self.archives_map:
                    word = self.archives_map[word]
                prob = self.initial_prob_dict[word]
                output_list.append([(prob, word)])
            else:
                if word in self.transition_prob_dict[sentence[index-1]]:
                    prob += self.transition_prob_dict[sentence[index-1]][word]
                    output_list.append([(prob, word)])
                elif 'NA' in self.transition_prob_dict[sentence[index-1]]:
                    prob += self.transition_prob_dict[sentence[index-1]]['NA']
                    output_list.append([(prob, word)])
                # else:
                # call API to add new word to the data
        
        while len(output_list) < self.chain_length:
            output_list.append(self.back_track(output_list[-1]))
        
        out_dict = self.pretty_output(output_list)
        return out_dict
 
    
    
    
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
    
pred = MCpredict(3)
test_data = open ('test_archive_1.txt', 'r', encoding='utf-8')    
output_file = open ('output_archive_proxy_units.txt', 'w', encoding='utf-8')
for line in test_data:
    output_list = pred.predict_seq(line)
    output_file.write(str(output_list) + '\n')

  