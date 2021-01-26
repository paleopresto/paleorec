# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 15:34:14 2021

@author: shrav
"""
import numpy as np
import pandas as pd
import math
import collections
import copy
import json

def fill_q0(in_dict, dict_type):
    '''
    Add initial probabilites for all the items in the dataset to intial probability dict
    eg. items in proxyObservationType, units, interpretation/variable and interpretation/variableDetail
    
    Parameters
    ----------
    in_dict : dict
        Initial probability dict - 
    dict_type : dict
        Iterate over this dict to add its values to the initial probability dict.

    Returns
    -------
    None.

    '''
    for key in dict_type.keys():
        if key not in in_dict:
            in_dict[key] = 0
        else:
            in_dict[key] = in_dict[key] + dict_type[key]

def calc_freq(dataframe_obj, col1, col2, ini_map):
    '''
    Calculate the frequency of items in col2 for each item in column 1.
    Conditional Probability of col2 given column 1

    Parameters
    ----------
    dataframe_obj : pandas dataframe
        Dataframe object containing training data.
    col1 : str
        Column for which data is being calculated.
    col2 : str
        Column whose count is being taken.
    ini_map : dict
        Contains all the items to be considered for the model.

    Returns
    -------
    counter_dict : dict
        Containing count for all the items that appear against each item in col1.

    '''
    counter_dict = {}
    for index, row in dataframe_obj.iterrows():
        if row[col1] in ini_map:
            if row[col1] not in counter_dict:
                counter_dict[row[col1]] = {}
                if row[col2] in ini_map:
                    counter_dict[row[col1]][row[col2]] = 1
            else:
                if row[col2] in ini_map:
                    if row[col2] not in counter_dict[row[col1]]:
                        counter_dict[row[col1]][row[col2]] = 1
                    else:
                        counter_dict[row[col1]][row[col2]] += 1
    return counter_dict

def add_extra_keys(all_keys, transition_matrix):
    '''
    Add missing items for transition from single key to all items in the dataset.

    Parameters
    ----------
    all_keys : set
        Contains all the items that should be in the transition dict for each item.
    transition_matrix : dict
        Transition dict object according to the chain type.

    Returns
    -------
    transition_mat : dict
        Updated dict after adding all the items in the transition dict for each item.

    '''
    transition_mat = copy.deepcopy(transition_matrix)
    for key, in_dict in transition_mat.items():
        in_dict_set = set(in_dict.keys())
        extra_keys = all_keys.difference(in_dict_set)
        for e_key in extra_keys:
            in_dict[e_key] = 0
    return transition_mat


def add_one_smoothing(transition_matrix):
    '''
    Add one smoothing to prevent the issue caused due to 0 transition probability from one item to the next.
    Convert counts to log probabilities
    Parameters
    ----------
    transition_matrix : dict
        Transition dict for all items.

    Returns
    -------
    transition_mat : dict
        Updated transition dict with log probabilities.

    '''
    transition_mat = copy.deepcopy(transition_matrix)
    for key, in_dict in transition_mat.items():
        len_in_dict = len(in_dict) + sum(list(in_dict.values()))
        for in_key, in_val in in_dict.items():
            in_dict[in_key] = math.log((in_val + 1)) - math.log(len_in_dict)
    return transition_mat



final_df = pd.read_csv('lipdverse_downsampled.csv')
final_df = final_df.replace(np.nan, 'NA', regex=True)

counter_archive = collections.Counter(final_df['archiveType'])
counter_proxy = collections.Counter(final_df['proxyObservationType'])
counter_units = collections.Counter(final_df['units'])
counter_int_var = collections.Counter(final_df['interpretation/variable'])
counter_int_det = collections.Counter(final_df['interpretation/variableDetail'])


# Creating initial probability dict
q0_chain1 = {key : value for key, value in counter_archive.items()}
q0_chain2 = {key : value for key, value in counter_archive.items()}


# Add all items to the initial probability dict
fill_q0(q0_chain1, counter_proxy)
fill_q0(q0_chain1, counter_units)

fill_q0(q0_chain2, counter_proxy)
fill_q0(q0_chain2, counter_int_var)
fill_q0(q0_chain2, counter_int_det)

# add-one smoothing for the initial probability dict
for key, value in q0_chain1.items():
    q0_chain1[key] += 1
    
for key, value in q0_chain2.items():
    q0_chain2[key] += 1
    
# downsample value for 'NA'
q0_chain2['NA'] = q0_chain2['SD']

# calculate log probabilities for initial probability dict
q0_total = sum(list(q0_chain1.values()))
for key, value in q0_chain1.items():
    q0_chain1[key] = math.log(value) - math.log(q0_total)
    
q0_total = sum(list(q0_chain2.values()))
for key, value in q0_chain2.items():
    q0_chain2[key] = math.log(value) - math.log(q0_total)
    
# creating transition probabilities dict
transition_matrix_chain1 = {}
transition_matrix_chain2 = {}

# counter for archive -> proxyObservationType
archive_inferredVarType_df = final_df.filter(['archiveType', 'proxyObservationType'], axis=1)
archive_proxy = calc_freq(archive_inferredVarType_df, 'archiveType', 'proxyObservationType', q0_chain1)

# counter for proxyObservationType -> units
proxy_units_df = final_df.filter(['proxyObservationType','units'], axis=1)
proxy_units = calc_freq(proxy_units_df, 'proxyObservationType','units', q0_chain1)

# counter for proxyObservationType -> interpretation/variable
proxy_int_df = final_df.filter(['proxyObservationType','interpretation/variable'], axis=1)
proxy_int = calc_freq(proxy_int_df, 'proxyObservationType','interpretation/variable', q0_chain2)

# counter for interpretation/variable -> interpretation/variableDetail
int_var_detail_df = final_df.filter(['interpretation/variable', 'interpretation/variableDetail'], axis=1)
int_var_detail = calc_freq(int_var_detail_df, 'interpretation/variable', 'interpretation/variableDetail', q0_chain2)

transition_matrix_chain1.update(archive_proxy)
transition_matrix_chain1.update(proxy_units)

transition_matrix_chain2.update(archive_proxy)
transition_matrix_chain2.update(proxy_int)
transition_matrix_chain2.update(int_var_detail)

# add-one smoothing for transition_probabilities matrix
q0_chain1_set = set(q0_chain1.keys())
transition_matrix_chain1 = add_extra_keys(q0_chain1_set, transition_matrix_chain1)
transition_matrix_chain1 = add_one_smoothing(transition_matrix_chain1)

q0_chain2_set = set(q0_chain2.keys())
transition_matrix_chain2 = add_extra_keys(q0_chain2_set, transition_matrix_chain2)
transition_matrix_chain2 = add_one_smoothing(transition_matrix_chain2)

archives_map = {'marine sediment': 'MarineSediment', 'lake sediment': 'LakeSediment', 'glacier ice': 'GlacierIce', 'documents': 'Documents', 'borehole': 'Rock', 'tree': 'Wood', 'bivalve': 'MollusckShell', 'coral': 'Coral', '': '', 'speleothem': 'Speleothem', 'sclerosponge': 'Sclerosponge', 'hybrid': 'Hybrid', 'Sclerosponge': 'Sclerosponge', 'Speleothem': 'Speleothem', 'Coral': 'Coral', 'MarineSediment': 'MarineSediment', 'LakeSediment': 'LakeSediment', 'GlacierIce': 'GlacierIce', 'Documents': 'Documents', 'Hybrid': 'Hybrid', 'MolluskShell': 'MolluskShell', 'Lake': 'Lake'}
model_dict = {'archive_types': list(counter_archive.keys()),'proxy_obs_types': list(counter_proxy.keys()),'units': list(counter_units.keys()),'int_var': list(counter_int_var.keys()),'int_var_det': list(counter_int_det.keys()), 'archives_map':archives_map, 'q0_chain1' : q0_chain1, 'q0_chain2' : q0_chain2, 'transition_matrix_chain1' : transition_matrix_chain1, 'transition_matrix_chain2' : transition_matrix_chain2}

# write model to file 
with open('model.txt', 'w') as json_file:
  json.dump(model_dict, json_file)