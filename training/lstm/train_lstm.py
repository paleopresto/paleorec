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
import getopt
import matplotlib
import matplotlib.pyplot as plt

from sys import platform as _platform

if _platform == "win32":
    sys.path.insert(1, '..\..\\')
else:
    sys.path.insert(1, '../../')
from utils import fileutils


device = None
epochs = 100
learning_rate = 0.01

options, remainder = getopt.getopt(sys.argv[1:], 'e:l:')
for opt, arg in options:
    if opt in ('-e'):
        try:
            epochs = int(arg.strip())
        except ValueError:
            sys.exit('Epoch has to be a positive integer.')
    elif opt in ('-l'):
        try:
            learning_rate = float(arg.strip())
        except ValueError:
            sys.exit('Learning Rate has to be a numeric value.')

def convert_dataframe_to_list(dataframe_obj):
    '''
    Method to return list of all the values in a single row separated by spaces from the dataframe.
    All values that were space separated before are converted to a single word.
    example. Sea Surface Temperature -> SeaSurfaceTemperature

    Parameters
    ----------
    dataframe_obj : pandas dataframe
        Dataframe contains the training data.

    Returns
    -------
    new_list : list
        List of input sentences.
    reference_dict : dict
        Mapping of the word to its space-stripped version used for training.

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
    '''
    Read training data into dataframe for training the model.
    The training data needs to be Label Encoded because LSTM only works with float data.
    Select only num_batches*seq_size*batch_size amount of data to work on.

    Parameters
    ----------
    train_file : string
        File path for the training data.
    batch_size : int
        Used to divide the training data into batches for training.
    seq_size : int
        Defines the sequence size for the training sentences.
    for_units : boolean, optional
        Flag to signify if model is training for the chain archiveType -> proxyObservationType -> units. The default is False.

    Returns
    -------
    int_to_vocab : dict
        Mapping of the Label Encoding int to text.
    vocab_to_int : dict
        Mapping of the Label Encoding text to int.
    n_vocab : int
        Size of the Label Encoding Dict.
    in_text : list
        Contains the input text for training.
    out_text : list
        Corresponding output for the input text.
    reference_dict : dict
        Mapping of the word to its space-stripped version used for training.

    '''

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
    '''
    Returns a batch each for the input sequence and the expected output word.

    Parameters
    ----------
    in_text : list
        Label Encoded strings of text.
    out_text : list
        Label Encoded Output for each each input sequence.
    batch_size : int
        Parameter to signify the size of each batch.
    seq_size : int
        Parameter to signify length of each sequence. In our case we are considering 2 chains, one of length 3 and the other of length 6.

    Yields
    ------
    list
        batch of input text sequence each of seq_size.
    list
        batch of output text corresponding to each input.

    '''
    
    num_batches = np.prod(in_text.shape) // (seq_size * batch_size)
    # Increment the loop by seq_size, because we have unique sequence and not a continuation.
    for i in range(0, num_batches * seq_size, seq_size):
        yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size]

def get_loss_and_train_op(net, lr=0.001):
    '''
    We are using CrossEntropy as a Loss Function for this RNN Model since this is a Multi-class classification kind of problem.
    

    Parameters
    ----------
    net : neural network instance
        Loss function is set for the Neural Network.
    lr : float, optional
        Defines the learning rate for the neural network. The default is 0.001.

    Returns
    -------
    criterion : Loss function instance
        Loss Function instance for the neural network.
    optimizer : Optimizing function instance
        Optimizer used for the neural network.

    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    return criterion, optimizer

def print_save_loss_curve(loss_value_list, chain):
    '''
    Method to save the plot for the training loss curve.

    Parameters
    ----------
    loss_value_list : list
        List with the training loss values.
    chain : str
        To differentiate between the proxyObservationTypeUnits chain from the proxyObservationType & interpretation/variable chain.

    Returns
    -------
    None.

    '''
    
    plt.plot(loss_value_list, label='Train Loss')
        
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.legend(loc='best')
    fig = plt.gcf()
    if not sys.stdin.isatty():    
        plt.show()
        
    timestr = time.strftime("%Y%m%d_%H%M%S")    
    final_path = loss_curve_path + chain + 'training_loss_e_' + str(epochs) + '_l_'+ str(learning_rate) + '_' + timestr + '.png'
    print('\nSaving the plot at ', final_path)
    fig.savefig(final_path, bbox_inches='tight')
    plt.close()

def train_RNN(int_to_vocab, vocab_to_int, n_vocab, in_text, out_text, seq_size, for_units = False):
    '''
    Method to train an lstm model on in_text and out_text.
    This method will save the model for the last epoch.
    
    Parameters
    ----------
    int_to_vocab : dict
        Mapping of the Label Encoding int to text.
    vocab_to_int : dict
        Mapping of the Label Encoding text to int.
    n_vocab : int
        Size of the Label Encoding Dict.
    in_text : list
        Contains the input text for training.
    out_text : list
        Corresponding output for the input text.
    for_units : boolean, optional
        Flag to signify if model is training for the chain archiveType -> proxyObservationType -> units. The default is False.

    Returns
    -------
    None.

    '''
    net = RNNModule(n_vocab, seq_size,
                    flags.embedding_size, flags.lstm_size)
    net = net.to(device)

    criterion, optimizer = get_loss_and_train_op(net, learning_rate)

    iteration = 0
    loss_value_list = []
    
    if for_units:
        print('\nTraining Data for the chain archiveType -> proxyObservationType -> proxyObservationTypeUnits\n')
    else:
        print('\nTraining Data for the chain archiveType -> proxyObservationType -> interpretation/variable -> interpretation/variableDetail -> inferredVariable -> inferredVarUnits\n')
        
    for e in range(epochs):
        batches = get_batches(in_text, out_text, flags.batch_size, seq_size)
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
                loss_value_list.append(loss_value)
                print('Epoch: {}/{}'.format(e, 200),
                      'Iteration: {}'.format(iteration),
                      'Loss: {}'.format(loss_value))
    timestr = time.strftime("%Y%m%d_%H%M%S")    
    if for_units:
        print('\nSaving the model file...')
        torch.save(net.state_dict(),model_file_path + 'model_lstm_units_'+timestr+'.pth')
    else:
        print('\nSaving the model file...')
        torch.save(net.state_dict(),model_file_path + 'model_lstm_interp_'+timestr+'.pth')
    
    print_save_loss_curve(loss_value_list, 'proxy_units_' if for_units else 'proxy_interp_')

                
def main():
    
    global int_to_vocab, vocab_to_int, n_vocab, in_text, out_text, device
    global int_to_vocab_u, vocab_to_int_u, n_vocab_u, in_text_u, out_text_u
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    int_to_vocab, vocab_to_int, n_vocab, in_text, out_text, reference_dict = get_data_from_file(
        flags.train_file, flags.batch_size, flags.seq_size)
    
    int_to_vocab_u, vocab_to_int_u, n_vocab_u, in_text_u, out_text_u, reference_dict_u = get_data_from_file(
        flags.train_file, flags.batch_size, flags.seq_size_u, for_units=True)
    
    timestr = time.strftime("%Y%m%d_%H%M%S")
    model_tokens = {'model_tokens' : int_to_vocab, 'model_tokens_u' : int_to_vocab_u, 'reference_dict': reference_dict, 'reference_dict_u': reference_dict_u}
    with open(model_file_path+'model_token_info_'+timestr+'.txt', 'w') as json_file:
          json.dump(model_tokens, json_file)

    # Train for archive -> proxyObservationType -> interpretation/variable -> interpretation/variableDetail -> inferredVariable -> inferredVarUnits
    train_RNN(int_to_vocab, vocab_to_int, n_vocab, in_text, out_text, flags.seq_size)

    # Train for archive -> proxyObservationType -> units
    train_RNN(int_to_vocab_u, vocab_to_int_u, n_vocab_u, in_text_u, out_text_u, flags.seq_size_u, for_units=True)
    

if _platform == "win32":
    data_file_dir = '..\..\data\csv\\'
    model_file_path = '..\..\data\model_lstm\\' 
    loss_curve_path = '..\..\data\loss\\'
else:
    data_file_dir = '../../data/csv/'
    model_file_path = '../../data/model_lstm/'
    loss_curve_path = '../../data/loss/'

train_path = fileutils.get_latest_file_with_path(data_file_dir, 'lipdverse_downsampled_*.csv')

flags = Namespace(
    train_file = train_path,
    seq_size_u=3,
    seq_size=6,
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