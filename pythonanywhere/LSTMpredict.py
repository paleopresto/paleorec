# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 12:13:12 2021

@author: shrav
"""

import torch
from argparse import Namespace
import json
import os
import glob

from RNNModule import RNNModule

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

class LSTMpredict:
        
    def __init__(self, model_file_path, mc_model_file_path, topk):
        
        
        flags = Namespace(
            seq_size=7,
            batch_size=16,
            embedding_size=64,
            lstm_size=64,
            gradients_norm=5,
            initial_words=['MarineSediment'],
            predict_top_k=5,
            checkpoint_path=model_file_path,
        )
        # PATH for model file
        PATH = get_latest_file_with_path(model_file_path, 'model_lstm_interp_*.pth')
        PATH_UNITS = get_latest_file_with_path(model_file_path, 'model_lstm_units_*.pth')
        MODEL_TOKEN_INFO_PATH = get_latest_file_with_path(model_file_path, 'model_token_info_*.txt')
        MODEL_CATEGORY_INFO_PATH = get_latest_file_with_path(mc_model_file_path, 'model_mc_*.txt')
            
        # Initialize device to load model onto
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.topk = topk
        
        # Read token info 
        with open(MODEL_TOKEN_INFO_PATH, 'r') as json_file:
            self.model_tokens = json.load(json_file)
            
        self.int_to_vocab = self.model_tokens['model_tokens']
        self.int_to_vocab = {int(k):v for k,v in self.int_to_vocab.items()}
        self.vocab_to_int = {v:k for k,v in self.int_to_vocab.items()}
        n_vocab = len(self.int_to_vocab)
        
        self.int_to_vocab_u = self.model_tokens['model_tokens_u']
        self.int_to_vocab_u = {int(k):v for k,v in self.int_to_vocab_u.items()}
        self.vocab_to_int_u = {v:k for k,v in self.int_to_vocab_u.items()}
        n_vocab_u = len(self.int_to_vocab_u)
        
        self.reference_dict = self.model_tokens['reference_dict']
        self.reference_dict_val = set(self.reference_dict.values())
        self.reference_dict_u = self.model_tokens['reference_dict_u']
        
        # Initialize the model for archive -> proxyObservationType -> interpretation/variable -> 
        #                                         interpretation/variableDetail -> inferredVariable -> inferredVarUnits
        self.model = RNNModule(n_vocab, flags.seq_size, flags.embedding_size, flags.lstm_size)
        self.model.load_state_dict(torch.load(PATH, map_location=self.device), strict=False)
        
        
        # Initialize the model for archive -> proxyObservationType -> units
        self.model_u = RNNModule(n_vocab_u, flags.seq_size, flags.embedding_size, flags.lstm_size)
        self.model_u.load_state_dict(torch.load(PATH_UNITS, map_location=self.device), strict=False)
        
        # Read file to get category names list information
        with open(MODEL_CATEGORY_INFO_PATH, 'r') as f:
            model_mc = json.load(f)    
            
        self.names_set = {0 : set(model_mc['archive_types']), 1: set(model_mc['proxy_obs_types']), 
                      2: set(model_mc['units']), 3: set(model_mc['int_var']), 4: set(model_mc['int_var_det']), 
                      5: set(model_mc['inf_var']), 6: set(model_mc['inf_var_units'])}
        for i in range(6):
            self.names_set[i] = {val.replace(' ', '') for val in self.names_set[i]}
        
        self.archives_map = model_mc['archives_map']

    
    def predict(self, device, net, words, vocab_to_int, int_to_vocab, names_set):
        '''
        Returns the list of top 5 predictions for the provided list of words using the model stored in net.
        The device is initialized to CPU for the purpose of the predictions.

        Parameters
        ----------
        device : torch.device
            Device type to signify 'cpu' or 'gpu'.
        net : torch.module
            DESCRIPTION.
        words : list
            List of strings used for predicting the next string in the list of words.
        vocab_to_int : dict
            Mapping of strings to int used to embed the input strings.
        int_to_vocab : dict
            Mapping of int to string used in the process of running the model predictions.
        names_set : dict
            Mapping of fieldType(example proxyObsType, interpretation/variable) and list of all the possible values the field can take.

        Returns
        -------
        list
            Top 5 recommendations for the next string in the sequence of words.

        '''
        
        net.eval()
        top_k = 10
        state_h, state_c = net.zero_state(1)
        state_h = state_h.to(device)
        state_c = state_c.to(device)
        for w in words:
            if w not in vocab_to_int:
                return []
            ix = torch.tensor([[vocab_to_int[w]]]).to(device)
            output, (state_h, state_c) = net(ix, (state_h, state_c))
        
        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        
        output = []
        for val in choices[0]:
            name = int_to_vocab[val]
            if name in names_set:
                output.append(name)
            if len(output) == self.topk:
                break
        
        return output
    
    def predictForSentence(self, sentence, isInferred = False):
        '''
        This method is used from the Flask Server Code API. 
        This method handles the initialization of the lstm model for the two different prediction chains we are using in our system.
        
        archiveType -> proxyObservationType -> units
        archiveType -> proxyObservationType -> interpretation/variable -> interpretation/variableDetial -> inferredVariable -> inferredVarUnits
        
        Depending on the length of the input sentence and the variableType, it chooses the output that will be returned to the server.
        
        If the variableType == measured
        then we will be considering the complete chain for prediction
        example: If sentence length = 1, it contains the archiveType and output = prediction for proxyObservationType
        example: If sentence length = 2, it contains the archiveType and proxyObservationType and output = units and interpretation/variable
        so on..
        
        If the variableType == inferred
        then we will be considering the top attributes in the chain from proxyObservationType to interpretation/variableDetail 
        to predict the top 5 inferredVariable as the output .
        
        Parameters
        ----------
        sentence : string
            Input sentence to predict the next field.
        isInferred : boolean, optional
            True if variableType == 'inferred'. The default is False.

        Returns
        -------
        dict
            Contains the result list of predictions as the value.
            Depending on the length of the input sentence and the variableType,
            the dict can contain one item corresponding to key '0' or two items corresponding to the two keys '0' and '1'.

        '''
        
        input_sent_list = sentence.strip().split(',')
        input_sent_list = [val.replace(' ', '') for val in input_sent_list]
        input_sent_list = [self.reference_dict.get(val, val) for val in input_sent_list]

        if isInferred and len(input_sent_list) <= 2:
            
            inferredVar = None
            if len(input_sent_list) == 2:
                inferredVar = input_sent_list[1]
                if inferredVar not in self.names_set[5]:
                    return {'0': []}
                del input_sent_list[1]
            while(len(input_sent_list) < 4):
                sentence = (',').join(input_sent_list)
                if len(input_sent_list) == 2:
                    input_sent_list.append(self.predictForSentence(sentence)['1'][0])
                else:
                    input_sent_list.append(self.predictForSentence(sentence)['0'][0])
                
            if inferredVar:
                input_sent_list.append(inferredVar)
            names_set_ind = len(input_sent_list) + 1 if len(input_sent_list) >= 2 else len(input_sent_list)
            results = self.predict(self.device, self.model, input_sent_list, self.vocab_to_int, self.int_to_vocab, self.names_set[names_set_ind])
            return {'0':results}
        
        
        names_set_ind = len(input_sent_list) + 1 if len(input_sent_list) >= 2 else len(input_sent_list)
        if len(input_sent_list) == 2:
            results_units =  self.predict(self.device, self.model_u, input_sent_list, self.vocab_to_int_u, self.int_to_vocab_u, self.names_set[len(input_sent_list)])
            results = self.predict(self.device, self.model, input_sent_list, self.vocab_to_int, self.int_to_vocab, self.names_set[names_set_ind])
            return {'0':results_units, '1':results}
        else:
            results = self.predict(self.device, self.model, input_sent_list, self.vocab_to_int, self.int_to_vocab, self.names_set[names_set_ind])
            return {'0':results}