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

# FOR WINDOWS
# sys.path.insert(1, '..\\')
# FOR LINUX
sys.path.insert(1, '../')
from utils import readLipdFileutils



# FOR WINDOWS
# input_path = '..\data\wiki_lipd_files\\'
# FOR LINUX
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

# pages2k_input_path = 'D://annotating_paleoclimate_data//lipdverse_data//PAGES2kv2/'
# tup = list(os.walk(pages2k_input_path))
# pages2k_root, pages2k_dir_names, pages2k_file_names = tup[0][0], tup[0][1], tup[0][2]

# temp12k_input_path = 'D://annotating_paleoclimate_data//lipdverse_data//Temp12k1_0_1/'
# tup = list(os.walk(temp12k_input_path))
# temp12k_root, temp12k_dir_names, temp12k_file_names = tup[0][0], tup[0][1], tup[0][2]

# palmod_input_path = 'D://annotating_paleoclimate_data//lipdverse_data//PalMod1_0_1/'
# tup = list(os.walk(palmod_input_path))
# palmod_root, palmod_dir_names, palmod_file_names = tup[0][0], tup[0][1], tup[0][2]

# iso2k_input_path = 'D://annotating_paleoclimate_data//lipdverse_data//iso2k1_0_0/'
# tup = list(os.walk(iso2k_input_path))
# iso2k_root, iso2k_dir_names, iso2k_file_names = tup[0][0], tup[0][1], tup[0][2]


read_files_list = []
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

timestr = time.strftime("%Y%m%d_%H%M%S")

# FOR WINDOWS
# table_com_data_path = r'..\csv\common_lipdverse_table_'+timestr+'.csv'
# inf_table_com_data_path = r'..\csv\common_lipdverse_inferred_'+timestr+'.csv'
# merged_data_path = r'..\csv\merged_common_lipdverse_inferred_'+timestr+'.csv'
# FOR LINUX
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