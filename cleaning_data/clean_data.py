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
wiki_file_names_list_copy = copy.deepcopy(wiki_file_names_list)

pages2k_root, pages2k_dir_names, pages2k_file_names, common_pages2k = None, None, None, None
temp12k_root, temp12k_dir_names, temp12k_file_names, common_temp12k = None, None, None, None

read_files_list = []
table_com, inf_table_com = None, None
discard_set = set()

ignore_wiki = False

user_input = input('Please enter \'Y\' if you would like to ignore the wiki files: ')
if user_input in {'Y', 'y', 'Yes', 'yes'}:
    ignore_wiki = True

options, remainder = getopt.getopt(sys.argv[1:], 'o:p:t:')
    
def add_to_read_files_list_wiki(root_name, dataset_files_list):
    '''
    This method updates the list of lipd files used for create the training data.
    It first checks if a given lipd file is present in the dataset_files_list
    if yes,then it will create a full path name using the provided root_name,
    else it will use the lipd file available from the wiki.

    Parameters
    ----------
    root_name : string
        Root Directory for the files passed in the dataset_files_list
    dataset_files_list : list
        List of files to be read and processed using utils.readLipdFileUtils

    Returns
    -------
    None.

    '''
    global discard_set    
    
    dataset_files_set = set(dataset_files_list)
    common_file_names = dataset_files_set.intersection(wiki_file_names)
    
    for filename in wiki_file_names_list_copy:
        if filename not in discard_set and '.lpd' in filename:
            if common_file_names and filename in common_file_names:
                discard_set.add(filename)
                filename = os.path.join(root_name, filename.strip())
                read_files_list.append(filename)

def add_to_read_files_list(root_name, dataset_files_list):
    '''
    This method updates the list of lipd files used for create the training data.
    It adds all the files passed in the dataset_files_list annotated with its complete file path to the read_files_list.
    
    Parameters
    ----------
    root_name : string
        Root Directory for the files passed in the dataset_files_list
    dataset_files_list : list
        List of files to be read and processed using utils.readLipdFileUtils

    Returns
    -------
    None.

    '''
    for filename in dataset_files_list:
        if '.lpd' in filename:
            filename = os.path.join(root_name, filename.strip())
            read_files_list.append(filename)

def walk_error_handler(exception_instance):
    '''
    Exception raised by os.walk for an incorrect path name

    Parameters
    ----------
    exception_instance : exception
        exception instance whose information will be displayed to the user.

    Returns
    -------
    None.

    '''
    print(exception_instance.args.split(',')[2])
    sys.exit('Please run the program again with valid input')

for opt, arg in options:
    if opt in ('-p'):
        tup = list(os.walk(arg))
        pages2k_root, pages2k_dir_names, pages2k_file_names = tup[0][0], tup[0][1], tup[0][2]
        if ignore_wiki:
            add_to_read_files_list(pages2k_root, pages2k_file_names)
        else:
            add_to_read_files_list_wiki(pages2k_root, pages2k_file_names)
    elif opt in ('-t'):
        tup = list(os.walk(arg))
        temp12k_root, temp12k_dir_names, temp12k_file_names = tup[0][0], tup[0][1], tup[0][2]
        if ignore_wiki:
            add_to_read_files_list(temp12k_root, temp12k_file_names)
        else:
            add_to_read_files_list_wiki(temp12k_root, temp12k_file_names)
    elif opt in ('-o'):
        paths_list = arg.split(',')
        for path in paths_list:
            tup = list(os.walk(path.strip(), onerror=walk_error_handler))
            root_name, dir_name, file_names = tup[0][0], tup[0][1], tup[0][2]
            if ignore_wiki:
                add_to_read_files_list(root_name, file_names)
            else:
                add_to_read_files_list_wiki(root_name, file_names)

def get_data_from_lipd():
    '''
    This passes the read_files_list to the readLipdFileutils which returns a dataframe with proxyObservationType chain and the inferredVariableType chain.
    
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
    cwd = os.getcwd()
    if not ignore_wiki:
        for line in wiki_file_names_list:
            if line not in discard_set and '.lpd' in line:
                root_name = os.path.abspath(wiki_root)
                line = os.path.join(root_name, line.strip())
                read_files_list.append(line)
               
    table_com, inf_table_com = readLipdFileutils.read_lipd_files_list(read_files_list)
    os.chdir(cwd)
    
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
        table_com_data_path = r'..\data\csv\common_lipdverse_table_'+timestr+'.csv'
        inf_table_com_data_path = r'..\data\csv\common_lipdverse_inferred_'+timestr+'.csv'
        merged_data_path = r'..\data\csv\merged_common_lipdverse_inferred_'+timestr+'.csv'
    else:
        table_com_data_path = r'../data/csv/common_lipdverse_table_'+timestr+'.csv'
        inf_table_com_data_path = r'../data/csv/common_lipdverse_inferred_'+timestr+'.csv'
        merged_data_path = r'../data/csv/merged_common_lipdverse_inferred_'+timestr+'.csv'
    
    table_com.to_csv(table_com_data_path, sep = ',', encoding = 'utf-8',index = False)
    inf_table_com.to_csv(inf_table_com_data_path, sep = ',', encoding = 'utf-8',index = False)
    
    table1 = copy.deepcopy(table_com)
    table2 = copy.deepcopy(inf_table_com)
    
    table1 = table1.filter(['filename','archiveType','proxyObservationType', 'units', 'interpretation/variable', 'interpretation/variableDetail'], axis=1)
    table2 = table2.filter(['filename','inferredVariable', 'inferredVarUnits'], axis=1)
    
    merged = pd.merge(table1, table2, on=["filename"],how='right')
    merged = merged.replace(np.nan, 'NA', regex=True)
    merged = merged[merged.archiveType != 'NA']
    
    table1 = copy.deepcopy(table_com)
    common_lipdverse_df = table1.filter(['archiveType','proxyObservationType', 'units', 'interpretation/variable', 'interpretation/variableDetail', 'inferredVariable', 'inferredVarUnits'], axis=1)
    
    merged_filter = merged.filter(['archiveType','proxyObservationType', 'units', 'interpretation/variable', 'interpretation/variableDetail', 'inferredVariable', 'inferredVarUnits'], axis=1)
    
    frames = [common_lipdverse_df, merged_filter]
    common_lipdverse_df = pd.concat(frames)
    
    common_lipdverse_df = common_lipdverse_df.replace(np.nan, 'NA', regex=True)
    
    common_lipdverse_df.to_csv(merged_data_path, sep = ',', encoding = 'utf-8',index = False)

if __name__ == "__main__":
    get_data_from_lipd()
    store_data_as_csv()