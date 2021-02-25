# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 11:57:17 2021

@author: shrav
"""

import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from collections import Counter
from argparse import Namespace
import json
from RNNModule import RNNModule
import sys
import time

# FOR WINDOWS
# sys.path.insert(1, '../../')
# FOR LINUX
sys.path.insert(1, '..\..\')
from utils import fileutils


device = None

def convert_dataframe_to_list(dataframe_obj):
    '''
    return : list of all the values in a single row separated by spaces from the dataframe.
             all values that were space separated before are converted to a single word 
             example. Sea Surface Temperature -> SeaSurfaceTemperature
    '''
    reference_dict = {}
    
    dataframe_obj = dataframe_obj.replace(np.nan, 'NA', regex=True)
    lipd_data_list = dataframe_obj.values.tolist()

    new_list = []
    for lis in lipd_data_list:
        for val in lis:
            reference_dict[val] = val.replace(" ", "")
        lis = [val.replace(" ", "") for val in lis]
        lis = (',').join(lis)
        new_list.append(lis)
    
    return new_list, reference_dict

def get_data_from_file(train_file, batch_size, seq_size, for_units = False):

    lipd_data = pd.read_csv(train_file)
    if for_units:
        lipd_data = lipd_data.filter(['archiveType', 'proxyObservationType', 'units'])
    else:
        lipd_data = lipd_data.filter(['archiveType', 'proxyObservationType', 'interpretation/variable', 'interpretation/variableDetail', 'inferredVariable', 'inferredVarUnits'])
    new_list, reference_dict = convert_dataframe_to_list(lipd_data)
    
    token_list = (',').join(new_list)

    text = token_list.split(',')

    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
    vocab_to_int = {w: k for k, w in int_to_vocab.items()}
    n_vocab = len(int_to_vocab)

    int_text = [vocab_to_int[w] for w in text]
    num_batches = int(len(int_text) / (seq_size * batch_size))
    in_text = int_text[:num_batches * batch_size * seq_size]
    out_text = np.zeros_like(in_text)
    out_text[:-1] = in_text[1:]
    out_text[-1] = in_text[0]
    in_text = np.reshape(in_text, (batch_size, -1))
    out_text = np.reshape(out_text, (batch_size, -1))


    return int_to_vocab, vocab_to_int, n_vocab, in_text, out_text, reference_dict

def get_batches(in_text, out_text, batch_size, seq_size):
    num_batches = np.prod(in_text.shape) // (seq_size * batch_size)
    # Increment the loop by seq_size, because we have unique sequence and not a continuation.
    for i in range(0, num_batches * seq_size, seq_size):
        yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size]

def get_loss_and_train_op(net, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    return criterion, optimizer

def train_RNN(int_to_vocab, vocab_to_int, n_vocab, in_text, out_text, for_units = False):
    net = RNNModule(n_vocab, flags.seq_size,
                    flags.embedding_size, flags.lstm_size)
    net = net.to(device)

    criterion, optimizer = get_loss_and_train_op(net, 0.01)

    iteration = 0
    
    
    for e in range(50):
        batches = get_batches(in_text, out_text, flags.batch_size, flags.seq_size)
        state_h, state_c = net.zero_state(flags.batch_size)
        
        # Transfer data to GPU
        state_h = state_h.to(device)
        state_c = state_c.to(device)
        for x, y in batches:
            iteration += 1
            
            # Tell it we are in training mode
            net.train()

            # Reset all gradients
            optimizer.zero_grad()
            
            x = torch.tensor(x).to(device).long()
            y = torch.tensor(y).to(device).long()

            logits, (state_h, state_c) = net(x, (state_h, state_c))
            loss = criterion(logits.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss_value = loss.item()

            # Perform back-propagation
            loss.backward()
            
            _ = torch.nn.utils.clip_grad_norm_(
                net.parameters(), flags.gradients_norm)

            # Update the network's parameters
            optimizer.step()
            
            if iteration % 100 == 0:
                print('Epoch: {}/{}'.format(e, 200),
                      'Iteration: {}'.format(iteration),
                      'Loss: {}'.format(loss_value))
    timestr = time.strftime("%Y%m%d_%H%M%S")    
    if for_units:    
        torch.save(net.state_dict(),model_file_path + 'model_lstm_units_'+timestr+'.pth')
    else:
        torch.save(net.state_dict(),model_file_path + 'model_lstm_interp_'+timestr+'.pth')
                
                
def main():
    
    global int_to_vocab, vocab_to_int, n_vocab, in_text, out_text, device
    global int_to_vocab_u, vocab_to_int_u, n_vocab_u, in_text_u, out_text_u
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    int_to_vocab, vocab_to_int, n_vocab, in_text, out_text, reference_dict = get_data_from_file(
        flags.train_file, flags.batch_size, flags.seq_size)
    
    int_to_vocab_u, vocab_to_int_u, n_vocab_u, in_text_u, out_text_u, reference_dict_u = get_data_from_file(
        flags.train_file, flags.batch_size, flags.seq_size, for_units=True)
    
    timestr = time.strftime("%Y%m%d_%H%M%S")
    model_tokens = {'model_tokens' : int_to_vocab, 'model_tokens_u' : int_to_vocab_u, 'reference_dict': reference_dict, 'reference_dict_u': reference_dict_u}
    with open(model_file_path+'model_token_info_'+timestr+'.txt', 'w') as json_file:
          json.dump(model_tokens, json_file)

    # Train for archive -> proxyObservationType -> interpretation/variable -> interpretation/variableDetail -> inferredVariable -> inferredVarUnits
    train_RNN(int_to_vocab, vocab_to_int, n_vocab, in_text, out_text)

    # Train for archive -> proxyObservationType -> units
    train_RNN(int_to_vocab_u, vocab_to_int_u, n_vocab_u, in_text_u, out_text_u, for_units=True)
    

# FOR WINDOWS
# data_file_dir = '..\..\data\csv\\'
# model_file_path = '..\..\data\model_lstm\\' 
# FOR LINUX
data_file_dir = '../../data/csv/'
model_file_path = '../../data/model_lstm/'

train_path = fileutils.get_latest_file_with_path(data_file_dir, 'lipdverse_downsampled_*.csv')

flags = Namespace(
    train_file = train_path,
    seq_size=7,
    batch_size=16,
    embedding_size=64,
    lstm_size=64,
    gradients_norm=5,
    initial_words=['MarineSediment'],
    predict_top_k=5,
    checkpoint_path=model_file_path,
)

new_list = []
reference_dict = {}
int_to_vocab = {}
vocab_to_int = {}
n_vocab_u = 0

if __name__ == '__main__':
    main()