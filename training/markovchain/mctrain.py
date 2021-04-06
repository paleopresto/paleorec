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
import itertools
import sys
import time

from sys import platform as _platform

if _platform == "win32":
    sys.path.insert(1, '..\..\\')
else:
    sys.path.insert(1, '../../')
from utils import fileutils

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

def calc_freq_multiple(dataframe_obj, ini_map, *argv):
    '''
    Calculate the frequency of items for all the columns in argv.
    Conditional Probability of last column given all the other columns except the last.

    Parameters
    ----------
    dataframe_obj : pandas dataframe
        Dataframe object containing training data.
    ini_map : dict
        Contains all the items to be considered for the model.
    *argv : list
        Contains the names for the columns that are being considered for calculating frequency.

    Returns
    -------
    counter_dict : dict
        Containing count for all the items that appear against each item in the last column.

    '''
    counter_dict = {}
    for index, row in dataframe_obj.iterrows():
        in_ini = True
        for arg in argv:
            if row[arg] not in ini_map:
                in_ini = False
                break
        
        if not in_ini:
            return None
        
        arg_ = ",".join([row[arg] for arg in argv[:-1]])
        last_ = row[argv[-1]]
        if arg_ not in counter_dict:
            counter_dict[arg_] = {}
            counter_dict[arg_][last_] = 1                    
        else:
            if last_ not in counter_dict[arg_]:
                    counter_dict[arg_][last_] = 1
            else:
                counter_dict[arg_][last_] += 1
    return counter_dict

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

if _platform == "win32":
    data_file_dir = '..\..\data\csv\\'
else:
    data_file_dir = '../../data/csv/'

data_file_path = fileutils.get_latest_file_with_path(data_file_dir, 'lipdverse_downsampled_*.csv')
final_df = pd.read_csv(data_file_path)
final_df = final_df.replace(np.nan, 'NA', regex=True)

counter_archive = collections.Counter(final_df['archiveType'])
counter_proxy = collections.Counter(final_df['proxyObservationType'])
counter_units = collections.Counter(final_df['units'])
counter_int_var = collections.Counter(final_df['interpretation/variable'])
counter_int_det = collections.Counter(final_df['interpretation/variableDetail'])
counter_inf_var = collections.Counter(final_df['inferredVariable'])
counter_inf_var_units = collections.Counter(final_df['inferredVarUnits'])


# Creating initial probability dict
q0_chain1 = {key : value for key, value in counter_archive.items()}
q0_chain2 = {key : value for key, value in counter_archive.items()}


# Add all items to the initial probability dict
fill_q0(q0_chain1, counter_proxy)
fill_q0(q0_chain1, counter_units)

fill_q0(q0_chain2, counter_proxy)
fill_q0(q0_chain2, counter_int_var)
fill_q0(q0_chain2, counter_int_det)
fill_q0(q0_chain2, counter_inf_var)
fill_q0(q0_chain2, counter_inf_var_units)

# add-one smoothing for the initial probability dict
for key, value in q0_chain1.items():
    q0_chain1[key] += 1
    
for key, value in q0_chain2.items():
    q0_chain2[key] += 1
    
# MANUAL STEP - TO SET THE FREQUENCY OF 'NA' TO BE THE MINIMUM SINCE WE DON'T START WITH 'NA'
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
# proxy_int_df = final_df.filter(['proxyObservationType','interpretation/variable'], axis=1)
# proxy_int = calc_freq(proxy_int_df, 'proxyObservationType','interpretation/variable', q0_chain2)

# counter for interpretation/variable -> interpretation/variableDetail
int_var_detail_df = final_df.filter(['interpretation/variable', 'interpretation/variableDetail'], axis=1)
int_var_detail = calc_freq(int_var_detail_df, 'interpretation/variable', 'interpretation/variableDetail', q0_chain2)

# counter for inferredVariable -> inferredVariableUnits
inf_var_units_df = final_df.filter(['inferredVariable', 'inferredVarUnits'], axis=1)
inf_var_units = calc_freq(inf_var_units_df, 'inferredVariable', 'inferredVarUnits', q0_chain2)

transition_matrix_chain1.update(archive_proxy)
transition_matrix_chain1.update(proxy_units)

transition_matrix_chain2.update(archive_proxy)
# transition_matrix_chain2.update(proxy_int)
transition_matrix_chain2.update(int_var_detail)
transition_matrix_chain2.update(inf_var_units)

# add-one smoothing for transition_probabilities matrix
q0_chain1_set = set(q0_chain1.keys())
transition_matrix_chain1 = add_extra_keys(q0_chain1_set, transition_matrix_chain1)
transition_matrix_chain1 = add_one_smoothing(transition_matrix_chain1)

q0_chain2_set = set(q0_chain2.keys())
transition_matrix_chain2 = add_extra_keys(q0_chain2_set, transition_matrix_chain2)
transition_matrix_chain2 = add_one_smoothing(transition_matrix_chain2)

# for interpretation/variable
# interpretation/variable = P[interpretation/variable | archiveType, proxyObservationType]

arch_pr_int_df = final_df.filter(['archiveType', 'proxyObservationType', 'interpretation/variable'])
arch_pr_int = calc_freq_multiple(arch_pr_int_df, q0_chain2, 'archiveType', 'proxyObservationType', 'interpretation/variable')

# check all the pairs of archive,proxyObservationType
combo_arch_proxy = list(itertools.product(list(counter_archive.keys()),list(counter_proxy.keys())))

for tup in combo_arch_proxy:
    if (',').join(tup) not in arch_pr_int:
        arch_pr_int[(',').join(tup)] = {}

arch_pr_int_chain2 = add_extra_keys(q0_chain2_set, arch_pr_int)
arch_pr_int_chain2 = add_one_smoothing(arch_pr_int_chain2)

for key, in_dict in arch_pr_int_chain2.items():
    arch, proxy = key.split(',')
    prob_arch_proxy = transition_matrix_chain2[arch][proxy]
    for in_key, in_val in in_dict.items():
        in_dict[in_key] = in_val - prob_arch_proxy
        
# for inferredVariable
# inferredVariable = P[inferredVariable | interpretation/variable, interpretation/variableDetail, proxyObservationType]
# to get Numerator = P(proxy intersect interpretation/variable intersect interpretation/variableDetail intersect inferredVariable)

for_inf_df = final_df.filter(['proxyObservationType', 'interpretation/variable', 'interpretation/variableDetail', 'inferredVariable'])
for_inf = calc_freq_multiple(for_inf_df, q0_chain2, 'proxyObservationType', 'interpretation/variable', 'interpretation/variableDetail', 'inferredVariable')

# check all the pairs of proxyObservationType, interpretation/variable, interpretation/variableDetail
combo_proxy_int_vardet = list(itertools.product(list(counter_proxy.keys()),list(counter_int_var.keys()), list(counter_int_det.keys())))

for tup in combo_proxy_int_vardet:
    if (',').join(tup) not in for_inf:
        for_inf[(',').join(tup)] = {}

pr_int_var_det_inf_chain2 = add_extra_keys(q0_chain2_set, for_inf)
pr_int_var_det_inf_chain2 = add_one_smoothing(pr_int_var_det_inf_chain2)

# to get Denominator = P(proxy intersect interpretation/variable intersect interpretation/variableDetail)
pr_int_var_detail_df = final_df.filter(['proxyObservationType', 'interpretation/variable', 'interpretation/variableDetail'])
pr_int_var_detail = calc_freq_multiple(pr_int_var_detail_df, q0_chain2, 'proxyObservationType', 'interpretation/variable', 'interpretation/variableDetail')

# check all the pairs of proxyObservationType, interpretation/variable, interpretation/variableDetail
combo_proxy_int_var = list(itertools.product(list(counter_proxy.keys()),list(counter_int_var.keys())))

for tup in combo_proxy_int_var:
    if (',').join(tup) not in pr_int_var_detail:
        pr_int_var_detail[(',').join(tup)] = {}

pr_int_var_det_chain2 = add_extra_keys(q0_chain2_set, pr_int_var_detail)
pr_int_var_det_chain2 = add_one_smoothing(pr_int_var_det_chain2)

for key, in_dict in pr_int_var_det_inf_chain2.items():
    pr, int_var, int_var_det = key.split(',')
    prob_proxy_int_var_det = pr_int_var_det_chain2[(',').join([pr, int_var])][int_var_det]
    for in_key, in_val in in_dict.items():
        in_dict[in_key] = in_val - prob_proxy_int_var_det


transition_matrix_chain2.update(arch_pr_int_chain2)
transition_matrix_chain2.update(pr_int_var_det_inf_chain2)

# inferredVariable = P[inferredVariable | interpretation/variable, interpretation/variableDetail, proxyObservationType]

# archives_map = {'marine sediment': 'MarineSediment', 'lake sediment': 'LakeSediment', 'glacier ice': 'GlacierIce', 'documents': 'Documents', 'borehole': 'Rock', 'tree': 'Wood', 'bivalve': 'MollusckShell', 'coral': 'Coral', '': '', 'speleothem': 'Speleothem', 'sclerosponge': 'Sclerosponge', 'hybrid': 'Hybrid', 'Sclerosponge': 'Sclerosponge', 'Speleothem': 'Speleothem', 'Coral': 'Coral', 'MarineSediment': 'MarineSediment', 'LakeSediment': 'LakeSediment', 'GlacierIce': 'GlacierIce', 'Documents': 'Documents', 'Hybrid': 'Hybrid', 'MolluskShell': 'MolluskShell', 'Lake': 'Lake', 'molluskshell': 'MollusckShell', 'Wood': 'Wood', 'Rock': 'Rock'}
# model_dict = {'archive_types': list(counter_archive.keys()),'proxy_obs_types': list(counter_proxy.keys()),'units': list(counter_units.keys()),'int_var': list(counter_int_var.keys()),'int_var_det': list(counter_int_det.keys()), 'inf_var': list(counter_inf_var.keys()), 'inf_var_units': list(counter_inf_var_units.keys()), 'archives_map':g['archives_map'], 'q0_chain1' : q0_chain1, 'q0_chain2' : q0_chain2, 'transition_matrix_chain1' : transition_matrix_chain1, 'transition_matrix_chain2' : transition_matrix_chain2}
model_dict = {'q0_chain1' : q0_chain1, 'q0_chain2' : q0_chain2, 'transition_matrix_chain1' : transition_matrix_chain1, 'transition_matrix_chain2' : transition_matrix_chain2}

# write model to file
timestr = time.strftime("%Y%m%d_%H%M%S")

if _platform == "win32":
    model_file_path = '..\..\data\model_mc\model_mc_'+timestr+'.txt'
else:
    model_file_path = '../../data/model_mc/model_mc_'+timestr+'.txt'
 
with open(model_file_path, 'w') as json_file:
  json.dump(model_dict, json_file)