# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 17:21:35 2021

@author: shrav
"""

import json
from heapq import heappop, heappush, heapify 

import os
import glob

def get_latest_file_with_path(path, *paths):
    '''
    Method to get the full path name for the latest file for the input parameter in paths.
    This method uses the os.path.getctime function to get the most recently created file that matches the filename pattern in the provided path. 

    Parameters
    ----------
    path : string
        Root pathname for the files.
    *paths : string list
        These are the var args field, the optional set of strings to denote the full path to the file names.

    Returns
    -------
    latest_file : string
        Full path name for the latest file provided in the paths parameter.

    '''

    fullpath = os.path.join(path, *paths)
    list_of_files = glob.iglob(fullpath)
    if not list_of_files:                
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

class MCpredict:

    def __init__(self, chain_length, top_k, model_file_path, ground_truth_path):
        '''
        Constructor to define object of Predict class.
        Thus we will have to read the model only once instead of having to read it every time we call the predict function.

        Parameters
        ----------
        chain_length : int
            Determine the type of prediction.
            There are 2 chain types:
            archive -> proxyObservationType -> units, 
            archive -> proxyObservationType -> interpretation/variable, interpretation/variableDetail, inferredVariable, inferredVarUnits

        Returns
        -------
        None.

        '''
        model_file_path = get_latest_file_with_path(model_file_path, 'model_mc_*.txt')
        with open(model_file_path, 'r') as f:
            model = json.load(f)
        
        ground_truth_path = get_latest_file_with_path(ground_truth_path, 'ground_truth_label_*.json')
        with open(ground_truth_path, 'r') as f:
            ground_truth = json.load(f)
         
        self.archives_map = ground_truth['archives_map']
        self.names_set = {0 : set(ground_truth['archive_types']), 1: set(ground_truth['proxy_obs_types']), 
                          2: set(ground_truth['units']), 3: set(ground_truth['int_var']), 4: set(ground_truth['int_var_det']), 
                          5: set(ground_truth['inf_var']), 6: set(ground_truth['inf_var_units'])}
        self.top_k = top_k
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
                if inner_list:
                    out_dict[str(i)] = inner_list
        
        return out_dict
      
        
    def get_max_prob(self, temp_names_set, trans_dict_for_word, prob):
        '''
        Find the maximimum items from a list stream using heapq.
        We will only pick those items that belong to the category we are interested in.
        Example : only recommend values in Units for Units.

        Parameters
        ----------
        temp_names_set : set
            Set containing the items in the category.
        trans_dict_for_word : dict
            Transition probability dict for the start word.
        prob : float
            The probability of the start word.

        Returns
        -------
        list
            Contains the top 5 recommendation for the start word.

        '''
        pq = []
        heapify(pq)
        
        for key, val in trans_dict_for_word.items():
            if key in temp_names_set:
                if len(pq) < self.top_k:
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
    
    
    def back_track(self, data, name_list_ind, sentence = None):
        '''
        Function to get top 5 items for each item in sequence
    
        Parameters
        ----------
        data : list/str
            Input sequence.
        name_list_ind: int
            Index for names_list dict. 
            Used to predict only proxyObservationType after Archive, 
            and not give recommendations from other category.
    
        Returns
        -------
        list
            Output list for the input sequence.
    
        '''
        outlist = []
        if type(data) is list:
            for content in data:
                outlist.append(self.back_track(content, name_list_ind, sentence))
        else:
            prob, word = data
            if name_list_ind == 3:
                word = (',').join([sentence[0], word])                 
            
            if word not in self.transition_prob_dict:
                return []
            temp_names_set = self.names_set[name_list_ind]
            return self.get_max_prob(temp_names_set, self.transition_prob_dict[word], prob)
        return outlist
    
    def get_ini_prob(self, sentence):
        '''
        Method to find the transition probability for the given sentence.
        For the first word we use the initial probability and for the rest of the sentence we use the transition probability for getting the next word.

        Parameters
        ----------
        sentence : str
            Input string sequence for which we have to predict the next sequence.

        Returns
        -------
        output_list : list
            Output list containing the probability and word for each stage of the sequence.
        sentence : list
            Sentence strip and split on space and returned for further use.

        '''
        
        sentence = sentence.strip().split(',')
        sentence = [x.strip() for x in sentence if x!='Select']
        output_list = []
        for index, word in enumerate(sentence):
            #initial probability
            prob = 0
            if index == 0:
                if word in self.archives_map:
                    word = self.archives_map[word]
                    sentence[0] = word
                prob = self.initial_prob_dict[word]
                output_list.append([(prob, word)])
                
            else:
                # print(sentence[index-1])
                dict_key = sentence[index-1]
                if index == 2:
                    dict_key = (',').join(sentence[:2])
                elif index == 4:
                    dict_key = (',').join(sentence[1:4])
                if word in self.transition_prob_dict[dict_key]:
                    prob += self.transition_prob_dict[dict_key][word]
                    output_list.append([(prob, word)])
                elif 'NA' in self.transition_prob_dict[dict_key]:
                    prob += self.transition_prob_dict[dict_key]['NA']
                    output_list.append([(prob, word)])
                # else:
                # call API to add new word to the data
        return (output_list, sentence)
    
    def predict_seq(self, sentence, isInferred = False):
        '''
        Predict the top 5 elements at each stage for every item in the chain
        There are 2 chain types:
            archive -> proxyObservationType -> units, 
            archive -> proxyObservationType -> interpretation/variable, interpretation/variableDetail
            ->inferredVariable -> inferredVarUnits
        
        We do not include inferredVariableType and inferredVarUnits in the sequential prediction, 
        but provide the recommendation after the interpretation/variableDetail has been selected.
        
        If isInferred == True, then we will choose the top value in prediction for the chain given the archiveType
        example:
            archiveType = MarineSediment
            proxy = D180
            interpretation/variable = NA
            interpretation/variableDetail = NA
        then based on this generate the top 5 predictions for inferredVariable
        
        
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
        final output: {'0': ['Trsgi', 'ARS', 'EPS', 'SD', 'Core']}
    
        '''
        sentence = sentence.strip()
        sent = sentence.split(',')
        sent = [x.strip() for x in sent if x!='Select']
        
        if isInferred and len(sent) <= 2:
            
            inferredVar = None
            if len(sent) == 2:
                inferredVar = sent[1]
                if inferredVar not in self.names_set[5]:
                    return {'0': []}
                del sent[1]
            
            sentence = (',').join(sent[:1])
            output_list, sent = self.get_ini_prob(sentence)
            
            for i in range(3):
                if len(output_list) >= 2:
                    result_list = self.back_track(output_list[-1], len(output_list)+1, sent)
                else:
                    result_list = self.back_track(output_list[-1], len(output_list))
                if result_list[0]:
                    output_list.append([result_list[0][0]])
                else:
                    break
                
            if inferredVar:
                prob = output_list[-1][0][0]
                temp_names_set = self.names_set[6]
                dict_key = (',').join(val[0][1] for val in output_list[1:])
                
                prob += self.transition_prob_dict[dict_key][inferredVar]
                output_list.append([(prob, inferredVar)])
                
                output_list.append(self.get_max_prob(temp_names_set, self.transition_prob_dict[dict_key], prob))
                
                out_dict = self.pretty_output(output_list)
                return {'0': out_dict['5']}
                
                
            if len(output_list) == 4:
                prob = output_list[-1][0][0]
                temp_names_set = self.names_set[5]
                dict_key = (',').join(val[0][1] for val in output_list[1:])
                output_list.append(self.get_max_prob(temp_names_set, self.transition_prob_dict[dict_key], prob))
            out_dict = self.pretty_output(output_list)
            return {'0': out_dict['4']}
        
        else:
            
            output_list, sent = self.get_ini_prob(sentence)
            
            prob = output_list[-1][0][0]
            if len(sent) == 4:
                temp_names_set = self.names_set[5]
                dict_key = sentence[sentence.index(',')+1:]
                output_list.append(self.get_max_prob(temp_names_set, self.transition_prob_dict[dict_key], prob))
            elif len(sent) == 5:
                temp_names_set = self.names_set[6]
                dict_key = sent[-1]
                output_list.append(self.get_max_prob(temp_names_set, self.transition_prob_dict[dict_key], prob))
            else :
                while len(output_list) < self.chain_length:            
                    if self.chain_length == 3:
                        output_list.append(self.back_track(output_list[-1], len(output_list)))
                    else:
                        if len(output_list) >= 2:
                            output_list.append(self.back_track(output_list[-1], len(output_list)+1, sent))
                        else:
                            output_list.append(self.back_track(output_list[-1], len(output_list)))   
        
        
        out_dict = self.pretty_output(output_list)
        return {'0': out_dict[str(len(sent))]}   
    
  