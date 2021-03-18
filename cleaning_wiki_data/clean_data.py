# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:32:53 2021

@author: shrav
"""

import pandas as pd
import numpy as np
import os
import time
import copy
import getopt
import sys
from sys import platform as _platform

if _platform == "win32":
    sys.path.insert(1, '..\\')
else:
    sys.path.insert(1, '../')
from utils import readLipdFileutils

if _platform == "win32":
    input_path = '..\data\wiki_lipd_files\\'
else:
    input_path = '../data/wiki_lipd_files/'

tup = list(os.walk(input_path))
wiki_root, wiki_dir_names, wiki_file_names_list = tup[0][0], tup[0][1], tup[0][2]
wiki_file_names = set(wiki_file_names_list)

pages2k_root, pages2k_dir_names, pages2k_file_names, common_pages2k = None, None, None, None
temp12k_root, temp12k_dir_names, temp12k_file_names, common_temp12k = None, None, None, None
palmod_root, palmod_dir_names, palmod_file_names, common_palmod = None, None, None, None
iso2k_root, iso2k_dir_names, iso2k_file_names, common_iso = None, None, None, None

options, remainder = getopt.getopt(sys.argv[1:], 'p:t:i:pm', ['pages2k, temp12k, iso2k, palmod'])

for opt, arg in options:
    if opt in ('-p', '--pages2k'):
        tup = list(os.walk(arg))
        pages2k_root, pages2k_dir_names, pages2k_file_names = tup[0][0], tup[0][1], tup[0][2]
        pages2k_file_names = set(pages2k_file_names)
        common_pages2k = pages2k_file_names.intersection(wiki_file_names)
    elif opt in ('-t', '--temp12k'):
        tup = list(os.walk(arg))
        temp12k_root, temp12k_dir_names, temp12k_file_names = tup[0][0], tup[0][1], tup[0][2]
        temp12k_file_names = set(temp12k_file_names)
        common_temp12k = temp12k_file_names.intersection(wiki_file_names)
    elif opt in ('-i', '--iso2k'):
        tup = list(os.walk(arg))
        palmod_root, palmod_dir_names, palmod_file_names = tup[0][0], tup[0][1], tup[0][2]
        palmod_file_names = set(palmod_file_names)
        common_palmod = palmod_file_names.intersection(wiki_file_names)
    elif opt in ('-pm', '--palmod'):
        tup = list(os.walk(arg))
        iso2k_root, iso2k_dir_names, iso2k_file_names = tup[0][0], tup[0][1], tup[0][2]
        iso2k_file_names = set(iso2k_file_names)
        common_iso = iso2k_file_names.intersection(wiki_file_names)

read_files_list = []
table_com, inf_table_com = None, None

def get_data_from_lipd():
    '''
    This method creates a list of lipd files to be read to create the training data.
    It first checks if a given lipd file is present in any of the lipdverse datasets, 
    if yes,then it will create a full path name using the root for any of the lipdverse datasets,
    else it will use the lipd file available from the wiki.
    
    This list of files is then passed to the readLipdFileutils which returns a dataframe with proxyObservationType chain and the inferredVariableType chain.
    
    Dataframes created
    table_com: pandas dataframe
        Contains information extracted for proxyObservationType        
    inf_table_com:pandas dataframe
        Contains information extracted for inferredVariableType

    Returns
    -------
    None.

    '''
    
    global table_com, inf_table_com
    
    for line in wiki_file_names_list:
        if '.lpd' in line:
            if common_pages2k and line in common_pages2k:
                root_name = pages2k_root
            elif common_temp12k and line in common_temp12k:
                root_name = temp12k_root
            elif common_palmod and line in common_palmod:
                root_name = palmod_root
            elif common_iso and line in common_iso:
                root_name = iso2k_root
            else:
                root_name = os.path.abspath(wiki_root)
            line = os.path.join(root_name, line.strip())
            read_files_list.append(line)
            
    table_com, inf_table_com = readLipdFileutils.read_lipd_files_list(read_files_list)

def store_data_as_csv():
    '''
    
    Given the dataframe for proxyObservationType(table_com) and the dataframe with the inferredVariableType(inf_table_com).
    This method merges the two dataframes to create a cleaned dataset for the provided data.

    Returns
    -------
        None.

    '''
    global table_com, inf_table_com
    
    timestr = time.strftime("%Y%m%d_%H%M%S")
    
    if _platform == "win32":
        table_com_data_path = r'..\csv\common_lipdverse_table_'+timestr+'.csv'
        inf_table_com_data_path = r'..\csv\common_lipdverse_inferred_'+timestr+'.csv'
        merged_data_path = r'..\csv\merged_common_lipdverse_inferred_'+timestr+'.csv'
    else:
        table_com_data_path = r'../csv/common_lipdverse_table_'+timestr+'.csv'
        inf_table_com_data_path = r'../csv/common_lipdverse_inferred_'+timestr+'.csv'
        merged_data_path = r'../csv/merged_common_lipdverse_inferred_'+timestr+'.csv'
    
    table_com.to_csv(table_com_data_path, sep = ',', encoding = 'utf-8',index = False)
    inf_table_com.to_csv(inf_table_com_data_path, sep = ',', encoding = 'utf-8',index = False)
    
    table1 = copy.deepcopy(table_com)
    table2 = copy.deepcopy(inf_table_com)
    
    table1 = table1.filter(['filename','archiveType','proxyObservationType', 'units', 'interpretation/variable', 'interpretation/variableDetail'], axis=1)
    table2 = table2.filter(['filename','inferredVariable', 'inferredVarUnits'], axis=1)
    
    merged = pd.merge(table1, table2, on="filename",how='right')
    
    table1 = copy.deepcopy(table_com)
    common_lipdverse_df = table1.filter(['archiveType','proxyObservationType', 'units', 'interpretation/variable', 'interpretation/variableDetail', 'inferredVariable', 'inferredVarUnits'], axis=1)
    
    merged_filter = merged.filter(['archiveType','proxyObservationType', 'units', 'interpretation/variable', 'interpretation/variableDetail', 'inferredVariable', 'inferredVarUnits'], axis=1)
    
    frames = [common_lipdverse_df, merged_filter]
    common_lipdverse_df = pd.concat(frames)
    
    common_lipdverse_df = common_lipdverse_df.replace(np.nan, 'NA', regex=True)
    
    common_lipdverse_df.to_csv(merged_data_path, sep = ',', encoding = 'utf-8',index = False)
    
    return 0

if __name__ == "__main__":
    get_data_from_lipd()
    store_data_as_csv()