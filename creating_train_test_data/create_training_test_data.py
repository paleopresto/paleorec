# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 15:39:08 2021

@author: shrav
"""

import pandas as pd
import numpy as np
import collections
from sklearn.utils import resample
import sys
import time
import json
from sys import platform as _platform
from pandas import ExcelWriter

if _platform == "win32":
    sys.path.insert(1, '..\\')
else:
    sys.path.insert(1, '../')
from utils import fileutils

common_lipdverse_df, final_df = None, None
names_set_dict = {}
counter_arch = {}
downsampled_df_train_list, downsampled_df_test_list= [], []
ground_truth_dict = {}
final_ground_truth_dict = {}


def read_latest_data_for_training():
    '''
    Method to read the latest file cleaned using utilities in cleaning_wiki_data/clean_data.py
    The latest data is picked up using the utilities file which uses os.ctime
    Reads the csv and stores the data into the common_lipdverse_df dataframe.
    
    Returns
    -------
    None.   

    '''
    global common_lipdverse_df
    
    # READ LATESTED MERGED LIPDVERSE DATA USING UTLITIES
    if _platform == "win32":
        data_file_dir = '..\data\csv\\'
    else:
        data_file_dir = '../data/csv/'
    
    data_file_path = fileutils.get_latest_file_with_path(data_file_dir, 'merged_common_lipdverse_inferred_*.csv')
    
    common_lipdverse_df = pd.read_csv(data_file_path)

def manually_clean_data_by_replacing_incorrect_values():
    '''
    Manual task to replace the following data in the dataframe with its alternative text.
    Could not eliminate these errors while reading lipd files using code in cleaning_wiki_files/clean_data.py
    Replace the data in place within the dataframe.
    
    Returns
    -------
    None.

    '''
    
    global common_lipdverse_df
    
    # MANUAL TASK - TO SCAN THROUGH DATA AND CHECK WHICH FIELDS ARE ACTUALLY SIMILAR BUT HAVE BEEN ENTERED INCORRECTLY
    common_lipdverse_df = common_lipdverse_df.replace(np.nan, 'NA', regex=True)
    common_lipdverse_df = common_lipdverse_df.replace('N/A', 'NA', regex=True)
    common_lipdverse_df = common_lipdverse_df.replace('Sr Ca', 'Sr/Ca', regex=True)
    common_lipdverse_df = common_lipdverse_df.replace('Depth_Cm', 'Depth', regex=True)
    common_lipdverse_df = common_lipdverse_df.replace('Depth_M', 'Depth', regex=True)
    common_lipdverse_df = common_lipdverse_df.replace('per mille VPDB', 'per mil VPDB', regex=True)
    common_lipdverse_df = common_lipdverse_df.replace('per mil VPBD', 'per mil VPDB', regex=True)
    common_lipdverse_df = common_lipdverse_df.replace('per mil', 'permil', regex=True)
    common_lipdverse_df = common_lipdverse_df.replace('g/cm^3', 'g/cm3', regex=True)
    common_lipdverse_df = common_lipdverse_df.replace('per cent', 'percent', regex=True)
    common_lipdverse_df = common_lipdverse_df.replace('µmol/mol', 'μmol/mol', regex=True)
    common_lipdverse_df = common_lipdverse_df.replace('mmolmol', 'mmol/mol', regex=True)
    common_lipdverse_df = common_lipdverse_df.replace('deg C', 'degC', regex=True)
    common_lipdverse_df = common_lipdverse_df.replace('Sea Surface D180W', 'Sea Surface D18Osw', regex=True)
    common_lipdverse_df = common_lipdverse_df.replace('d18ow','D18Osw', regex=True)
    common_lipdverse_df = common_lipdverse_df.replace('sed rate','Sedimentation Rate', regex=True)
    common_lipdverse_df = common_lipdverse_df.replace('d18ocorr','D18Ocorr', regex=True)
    common_lipdverse_df = common_lipdverse_df.replace('Thisshouldntbeempty','NA', regex=True)
    common_lipdverse_df = common_lipdverse_df.replace(',', '', regex=True)
    common_lipdverse_df = common_lipdverse_df.replace('unknown', 'NA', regex=True)
    common_lipdverse_df = common_lipdverse_df.replace('NotApplicable', 'NA', regex=True)

def write_autocomplete_data_file():
    '''
    Writes the data to autocomplete_file used for autocomplete suggestions on the UI

    Returns
    -------
    None.

    '''
    
    global names_set_dict
    
    timestr = time.strftime("%Y%m%d_%H%M%S")
    if _platform == "win32":
        autocomplete_file_path = '..\\data\\autocomplete\\autocomplete_file_'+timestr+'.json'
    else:
        autocomplete_file_path = '../data/autocomplete/autocomplete_file_'+timestr+'.json'
    
    with open(autocomplete_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(names_set_dict, json_file)

def take_user_input():
    '''
    Method to take user input to eliminate any-co-k occurance of data.
    This method validates the input to be a positive integer and returns it.

    Returns
    -------
    int
        Value for any-co-k elimination.

    '''
    
    k = input('Please enter the value of \'k\' to replace any-co-k instances : ')
    try:
        k = int(k.strip())
    except ValueError:
        sys.exit('Please enter an integer value to elimate any-co-k')
    return k if k > 0 else 0

def editDistDP(str1, str2, m, n):
    '''
    Calculates the edit distance between str1 and str2.

    Parameters
    ----------
    str1 : string
        Input string 1.
    str2 : TYPE
        Input string 2.
    m : int
        len of string 1
    n : int
        len of string 2

    Returns
    -------
    int
        Edit distance value between str1 and str2.

    '''
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]
 
    for i in range(m + 1):
        for j in range(n + 1):
 
            if i == 0:
                dp[i][j] = j
 
            elif j == 0:
                dp[i][j] = i
 
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
 
            else:
                dp[i][j] = 1 + min(dp[i][j-1],  
                                   dp[i-1][j],  
                                   dp[i-1][j-1])
 
    return dp[m][n]

def discard_less_frequent_values_from_data():
    '''
    This method reduces the subset of data to the fields in the chain, 
    i.e archiveType, proxyObservationType, units, interpretation/variable, interpretation/variableDetail, inferredVariable, inferredVarUnits.
    
    Create a dict to store autocomplete information for each fieldType.
    
    Generate a counter for the values in each column to understand the distribution of each individual field.
    Manually decide whether to eliminate any co 1 values within each field by taking user input.
    
    Update the dataframe by discarding values from there.

    Returns
    -------
    None.

    '''
    global final_df, names_set_dict, counter_arch, final_ground_truth_dict
    
    
    final_df = common_lipdverse_df.filter(['filename','archiveType','proxyObservationType', 'units', 'interpretation/variable', 'interpretation/variableDetail', 'inferredVariable', 'inferredVarUnits'], axis=1)
    
    archives_map = {"marine sediment": "MarineSediment", "lake sediment": "LakeSediment", "glacier ice": "GlacierIce", "documents": "Documents", "borehole": "Rock", "tree": "Wood", "bivalve": "MolluskShell", "mollusk shell": "MolluskShell", "coral": "Coral", "speleothem": "Speleothem", "sclerosponge": "Sclerosponge", "hybrid": "Hybrid", "Sclerosponge": "Sclerosponge", "Speleothem": "Speleothem", "Coral": "Coral", "MarineSediment": "MarineSediment", "LakeSediment": "LakeSediment", "GlacierIce": "GlacierIce", "Documents": "Documents", "Hybrid": "Hybrid", "MolluskShell": "MolluskShell", "Lake": "Lake", "molluskshell": "MolluskShell", "Wood": "Wood", "Rock": "Rock", "MollusckShell": "MolluskShell", "MolluskShells": "MolluskShell", "TerrestrialSediment": "TerrestrialSediment", "Midden": "Midden", "Peat": "Peat", "GroundIce": "GroundIce", "Ice-other": "Ice-other", "Marine Sediment" : "MarineSediment", "Lake Sediment" : "LakeSediment", "Mollusk Shell" : "MolluskShell", "Glacier Ice" : "GlacierIce", "Ground Ice" : "GroundIce", "Terrestrial Sediment" : "TerrestrialSediment"}
    new_archives = set()
    for i, row in final_df.iterrows():
        if row[1] in archives_map: 
            final_df.at[i,'archiveType'] = archives_map[row[1]]
        else:
            for w in archives_map.keys():
                if editDistDP(w, row[1], len(w), len(row[1])) <=5:
                    final_df.at[i,'archiveType'] = archives_map[w]
                else:
                    new_archives.add(row[1])
        if row[6] == 'Na':
            final_df.at[i,'interpretation/variableDetail'] = 'NA'
        if row[5] in {'Pdo', 'Amo', 'Pdsi'}:
            final_df.at[i,'interpretation/variable'] = row[5].upper()
            
    for arch in new_archives:
        archives_map[arch] = arch
    if 'NA' in archives_map:
        del archives_map['NA']
    final_ground_truth_dict['archives_map'] = archives_map

    final_df = final_df[final_df.units != 'Mg/Ca']
    
    counter_arch = collections.Counter(final_df['archiveType'])
    counter_proxy = collections.Counter(final_df['proxyObservationType'])
    counter_units = collections.Counter(final_df['units'])
    counter_int_var = collections.Counter(final_df['interpretation/variable'])
    counter_int_det = collections.Counter(final_df['interpretation/variableDetail'])
    counter_inf_var = collections.Counter(final_df['inferredVariable'])
    counter_inf_var_units = collections.Counter(final_df['inferredVarUnits'])
    
    # CODE TO WRITE INITIAL VALUES TO DIFFERENT SHEETS IN EXCEL
    writer = pd.ExcelWriter("label_correction.xlsx", engine='xlsxwriter')

    arch_df = pd.DataFrame.from_dict(counter_arch, orient='index').reset_index()
    arch_df.to_excel(writer, sheet_name='archiveTypes')
    
    pr_df = pd.DataFrame.from_dict(counter_proxy, orient='index').reset_index()
    pr_df.to_excel(writer, sheet_name='proxyObsType')
    
    units_df = pd.DataFrame.from_dict(counter_units, orient='index').reset_index()
    units_df.to_excel(writer, sheet_name='proxyUnits')
    
    int_df = pd.DataFrame.from_dict(counter_int_var, orient='index').reset_index()
    int_df.to_excel(writer, sheet_name='interpVariable')
    
    int_det_df = pd.DataFrame.from_dict(counter_int_det, orient='index').reset_index()
    int_det_df.to_excel(writer, sheet_name='interpVarDet')
    
    inf_df = pd.DataFrame.from_dict(counter_inf_var, orient='index').reset_index()
    inf_df.to_excel(writer, sheet_name='inferredVariable')
    
    inf_units_df = pd.DataFrame.from_dict(counter_inf_var_units, orient='index').reset_index()
    inf_units_df.to_excel(writer, sheet_name='inferredVarUnits')

    writer.save()
    
    # Add to a file for autocomplete suggestions without removing any co 1
    names_set_dict = {'proxyObservationType' : list(counter_proxy.keys()), 'proxyObservationTypeUnits' : list(counter_units.keys()), 
                      'interpretation/variable' : list(counter_int_var.keys()), 'interpretation/variableDetail' : list(counter_int_det.keys()), 
                      'inferredVariable' : list(counter_inf_var.keys()), 'inferredVariableUnits' : list(counter_inf_var_units.keys())}
    write_autocomplete_data_file()
         
    # PROXY OBSERVATION TYPE
    print('\nSamples per instance of proxyObservationType : ',counter_proxy)
    k = take_user_input()  
    counter_proxy_a = {key:value for key,value in dict(counter_proxy).items() if value > k}
    
    # INTERPRETATION/VARIABLE
    print('\nSamples per instance of interpretation/variable : ',counter_int_var)
    k = take_user_input()
    counter_int_var_a = {key:value for key,value in dict(counter_int_var).items() if value > k}
    
    # INTERPRETATION/VARIABLE DETAIL
    print('\nSamples per instance of interpretation/variableDetail: ',counter_int_det)
    k = take_user_input()
    counter_int_det_a = {key:value for key,value in dict(counter_int_det).items() if value > k}
    
    # discard set for proxyObservationType
    discard_set = set(counter_proxy.keys()).difference(set(counter_proxy_a.keys()))
    discard_set.add('Depth')
    final_df = final_df[~final_df['proxyObservationType'].isin(discard_set)]
    
    # discard set for interpretation variable
    discard_set = set(counter_int_var.keys()).difference(set(counter_int_var_a.keys()))
    final_df = final_df[~final_df['interpretation/variable'].isin(discard_set)]
    
    # discard set for interpretation variable detail
    discard_set = set(counter_int_det.keys()).difference(set(counter_int_det_a.keys()))
    final_df = final_df[~final_df['interpretation/variableDetail'].isin(discard_set)]    
    
    # MANUAL TASK - REMOVE INFERRED VARIABLE TYPES RELATED TO AGE AND OTHER MEASURED VARIABLE TYPE VALUES ENTERED IN THE WIKI
    # discard set for inferredVariableType
    discard_set = {'Calendar Age', 'Accumulation rate', 'k37 n toc', 'acc rate k37', 'acc rate toc', 'Age', 'Radiocarbon Age', 'mg/ca',' Year'}
    final_df = final_df[~final_df['inferredVariable'].isin(discard_set)]
    
    # DROP NA VALUES FROM THE ARCHIVE TYPE FIELD AS IT CAN NEVER CONTAIN NA
    final_df = final_df[final_df.archiveType != 'NA']
    final_df.dropna(subset=['archiveType'], inplace=True)

def get_label_set_for_input(dataframe_obj, col1, col2):
    '''
    Calculate the items in col2 for each item in column 1.
    

    Parameters
    ----------
    dataframe_obj : pandas dataframe
        Dataframe object containing training data.
    col1 : str
        Column for which data is being calculated.
    col2 : str
        Column whose item is being taken.
    
    Returns
    -------
    counter_dict : dict
        Containing set of all the items that appear against each item in col1.

    '''
    counter_dict = {}
    for index, row in dataframe_obj.iterrows():
        if row[col1] not in counter_dict:
            counter_dict[row[col1]] = set()
            counter_dict[row[col1]].add(row[col2])
        else:
            if row[col2] not in counter_dict[row[col1]]:
                counter_dict[row[col1]].add(row[col2])

    return counter_dict

def update_ground_truth_dict(temp_dict):
    '''
    Method to add values from one dict to another. 
    If a key is present append the list of values to the already created list.

    Parameters
    ----------
    temp_dict : dict
        Dict whose values need to be added to the ground truth dict.

    Returns
    -------
    None.

    '''
    global ground_truth_dict

    for k, v in temp_dict.items():
        if k in ground_truth_dict:
            ground_truth_dict[k].union(v)
        else:
            ground_truth_dict[k] = v

def generate_ground_truth_label_info(final_df_test):
    '''
    Method to collect the list of all possible next values for a given field.
    Example:
        Given Marine Sediment
        Ouput for Proxy Observation Type  = ["Notes", "Mg/Ca", "Bsi", "Caco3", "Uk37", "Mgca", "IRD", "D18O", "37:2Alkenoneconcentration", "TOC", "D18O.Error", "DBD", "D13C", "Dd", "D13C.Error", "Foram.Abundance"]

    Parameters
    ----------
    final_df_test : pandas dataframe
        Final Dataframe on which Counts for the various fields are calculated.

    Returns
    -------
    None.

    '''
    global ground_truth_dict, final_ground_truth_dict

    archive_proxy = get_label_set_for_input(final_df_test, 'archiveType', 'proxyObservationType')
    update_ground_truth_dict(archive_proxy)

    proxy_units = get_label_set_for_input(final_df_test, 'proxyObservationType','units')

    proxy_int = get_label_set_for_input(final_df_test, 'proxyObservationType','interpretation/variable')
    update_ground_truth_dict(proxy_int)

    int_var_detail = get_label_set_for_input(final_df_test, 'interpretation/variable', 'interpretation/variableDetail')
    update_ground_truth_dict(int_var_detail)

    inf_var = get_label_set_for_input(final_df_test, 'interpretation/variableDetail','inferredVariable')
    update_ground_truth_dict(inf_var)

    inf_var_units = get_label_set_for_input(final_df_test, 'inferredVariable', 'inferredVarUnits')
    update_ground_truth_dict(inf_var_units)

    ground_truth_dict = {k:list(v) for k,v in ground_truth_dict.items()}
    ground_truth_dict['proxyUnits'] = {k:list(v) for k,v in proxy_units.items()}
    final_ground_truth_dict['ground_truth'] = ground_truth_dict


def calculate_counter_info(final_df):
    '''
    Method to get list of all possible values for each fields in the recommendation system.

    Parameters
    ----------
    final_df : pandas dataframe
        Pandas dataframe consisting of training information.

    Returns
    -------
    None.

    '''

    global final_ground_truth_dict

    counter_archive = collections.Counter(final_df['archiveType'])
    counter_proxy = collections.Counter(final_df['proxyObservationType'])
    counter_units = collections.Counter(final_df['units'])
    counter_int_var = collections.Counter(final_df['interpretation/variable'])
    counter_int_det = collections.Counter(final_df['interpretation/variableDetail'])
    counter_inf_var = collections.Counter(final_df['inferredVariable'])
    counter_inf_var_units = collections.Counter(final_df['inferredVarUnits']) 

    final_ground_truth_dict.update({'archive_types': list(counter_archive.keys()),'proxy_obs_types': list(counter_proxy.keys()),'units': list(counter_units.keys()),'int_var': list(counter_int_var.keys()),'int_var_det': list(counter_int_det.keys()), 'inf_var': list(counter_inf_var.keys()), 'inf_var_units': list(counter_inf_var_units.keys())}) 

def downsample_archive(archiveType, downsample_val):
    '''
    
    Method to downsample an archiveType to the provided value in the params.
    This module also generates the test data for the given archiveType.

    Parameters
    ----------
    archiveType : str
        Archive Type to downsample.
    downsample_val : int
        Number of samples the archiveType needs to be reduced to.

    Returns
    -------
    None.

    '''
    global downsampled_df_train_list, downsampled_df_test_list
    
    df_arch = final_df[final_df.archiveType==archiveType]
    
    df_arch_downsampled = resample(df_arch, 
                                     replace=False,    # sample without replacement
                                     n_samples=downsample_val,
                                     random_state=27,  # reproducibility
                                     stratify=df_arch)
    
    # Add all unique values from the df_wood and df_marine_sed data frame into the downsampled dataframes.
    # We intend to provide our model all the unique values that are currently present in the data
    df_arch_nodup = df_arch.drop_duplicates()
    df_arch_ds_no_dup = df_arch_downsampled.drop_duplicates()
    
    df_arch_extra = df_arch_nodup.merge(df_arch_ds_no_dup, how='left', indicator=True)
    df_arch_extra = df_arch_extra[df_arch_extra['_merge']=='left_only']
    df_arch_extra = df_arch_extra.drop(columns=['_merge'])
    
    df_arch_downsampled = df_arch_downsampled.append(df_arch_extra, ignore_index=True)
    
    test_set_size_arch = downsample_val//5
    # df_arch_test = resample(df_arch_downsampled, 
    #                         replace=False,    # sample without replacement
    #                         n_samples=test_set_size_arch,     # to match minority class
    #                         random_state=123,  # reproducibility
    #                         stratify=df_arch_downsampled)
    df_arch_test = df_arch_downsampled.sample( 
                            replace=False,    # sample without replacement
                            frac=0.2,     # to match minority class
                            random_state=2021)  # reproducibility
    
    df_arch_downsampled = df_arch_downsampled.drop(df_arch_test.index)

    downsampled_df_train_list.append(df_arch_downsampled)
    downsampled_df_test_list.append(df_arch_test)
    
    if archiveType == 'Wood':
        # MANUAL TASK - ADD DATA FOR WOOD FROM INFERRED VARIABLE TYPE CSV FILE,
        # BECAUSE THERE ARE NO SAMPLES WITH UNITS FOR INFERRED VARIABLE TYPE AND INFERRED VARIABLE TYPE UNITS FOR ARCHIVE = WOOD
                
        if _platform == "win32":
            wood_inferred_path = '..\data\csv\wood_inferred_data.csv'
        else:
            wood_inferred_path = '../data/csv/wood_inferred_data.csv'
        
        try:
            wood_inferred_df = pd.read_csv(wood_inferred_path)
        except RuntimeError:
            print("Couldn't find the required file : wood_inferred_data.csv")
        wood_inferred_df = wood_inferred_df.replace(np.nan, 'NA', regex=True)
        
        wood_inferred_test = resample(wood_inferred_df, 
                                         replace=False,    # sample without replacement
                                         n_samples=2,     # to match minority class
                                         random_state=123)  # reproducibility
        downsampled_df_train_list.append(wood_inferred_df)
        downsampled_df_test_list.append(wood_inferred_test)
        

def downsample_archives_create_final_train_test_data():
    '''
    
    Manually decide based on the counter for archiveTypes which archiveTypes need to be downsampled.
    
    Two approaches were tried for creating the test data from the generated data.
    
    First was creating a test dataset by resampling from the training data.
    Since we do not even distribution of data across each class, we have used 'stratify' during resample.
    This will help us even out the distribution of data across all classess in the provided dataset.
    
    Second approach is to keep aside 20% of the generated data as unseen test data, while 80% of the data would be used as the training data.
    Using a bar plot distribution tried to verify the ratio of the archives to proxyObservationType in the training and test data are nearly equal.
    
    After the final training data is procured, the ground truth data file is created which is used in the final prediction.
    
    Returns
    -------
    None.

    '''
    global downsampled_df_train_list, downsampled_df_test_list, counter_arch
    
    # MANUAL TASK - DECIDE WHICH ARCHIVES NEED TO BE DOWN-SAMPLED
    counter_arch = dict(collections.Counter(final_df['archiveType']))
    if 'NA' in counter_arch:
        del counter_arch['NA']
    print('\nCount for each instance of Archive Type: ')
    cnt = 1
    counter_arch_list = [(key,val) for key,val in dict(counter_arch).items()]
    for key, value in counter_arch_list:
        print('{}. {} = {}'.format(str(cnt), key, str(value)))
        cnt += 1
    
    discard_set = set()
    archives = input('\nUsing the item number beside the archive type, list the archive types to be downsampled separated by \',\' or enter none: ') 
    archives = archives.lower()
    if archives != 'none':
        
        archives_list = archives.split(',')
        for i,arch_num in enumerate(archives_list):
            arch_num = arch_num.strip()
            discard_set.add(counter_arch_list[int(arch_num)-1][0])
            archives_list[i] = counter_arch_list[int(arch_num)-1][0]
            
        downsampled = input('\nPlease enter the numeric value to downsampled the above list of Archive Types in same order :')
        downsampled_list = downsampled.split(',')
        for i, n in enumerate(downsampled_list):
            try:
                num = int(n.strip())
                
                if num > counter_arch[archives_list[i]]:
                    print('The downsample value provided; {} is greater than the samples available for the archive Type; {}'.format(num, counter_arch[archives_list[i]]))
                    print('Going ahead with current number of samples.')
                    downsample_archive(archives_list[i], counter_arch[archives_list[i]])
                else:
                    downsample_archive(archives_list[i], num)
            except ValueError:
                print("Error! {} is not a number.".format(n))
                sys.exit('Please run the program with a valid integer.')
    
    df_rest = final_df[~final_df['archiveType'].isin(discard_set)]

    # creating graph for count of datasets for proxyObsType and InfVarType 

    # Code to get data for proxyObsType and InferredVar
    # dataset_proxy = get_label_set_for_input(df_rest, 'proxyObservationType', 'filename')
    # for k,v in dataset_proxy.items():
    #     dataset_proxy[k] = len(v)
    #     print('{} = {}'.format(k, str(len(v))))

    # print('*****************************************************')
    # dataset_infvar = get_label_set_for_input(df_rest, 'inferredVariable', 'filename')
    # for k,v in dataset_infvar.items():
    #     dataset_infvar[k] = len(v)
    #     print('{} = {}'.format(k, str(len(v))))
    


    df_rest = df_rest.filter(['archiveType','proxyObservationType', 'units', 'interpretation/variable', 'interpretation/variableDetail', 'inferredVariable', 'inferredVarUnits'], axis=1)
    df_rest_test = df_rest.sample( 
                            replace=False,    # sample without replacement
                            frac=0.2,     # to match minority class
                            random_state=2021)  # reproducibility
    df_rest = df_rest.drop(df_rest_test.index)
    
    downsampled_df_train_list.append(df_rest)
    downsampled_df_test_list.append(df_rest_test)
    
    final_df_downsampled = pd.concat(downsampled_df_train_list)
    final_df_test = pd.concat(downsampled_df_test_list)
    
    final_df_downsampled = final_df_downsampled.sample(frac=1, random_state=2021).reset_index(drop=True)
    final_df_test = final_df_test.sample(frac=1, random_state=2021).reset_index(drop=True)
    
    generate_ground_truth_label_info(final_df_test)
    calculate_counter_info(final_df_downsampled)

    timestr = time.strftime("%Y%m%d_%H%M%S")
    
    if _platform == "win32":
        lipd_downsampled_path = '..\data\csv\lipdverse_downsampled_'+timestr+'.csv'
        lipd_test_path = '..\data\csv\lipdverse_test_'+timestr+'.csv'
        ground_truth_path = '..\data\ground_truth_info\ground_truth_label_'+timestr+'.json'
    else:
        lipd_downsampled_path = '../data/csv/lipdverse_downsampled_'+timestr+'.csv'
        lipd_test_path = '../data/csv/lipdverse_test_'+timestr+'.csv'
        ground_truth_path = '../data/ground_truth_info/ground_truth_label_'+timestr+'.json'
    
    # write back the final training data to create the model.
    print('\n Creating file {} at location {}'.format('lipdverse_downsampled_'+timestr+'.csv', lipd_downsampled_path[:lipd_downsampled_path.rindex('lipdverse_down')]))
    final_df_downsampled.to_csv(lipd_downsampled_path, sep = ',', encoding = 'utf-8',index = False)
    # write back the final test data to calculate accuracy of the model.
    print('\n Creating file {} at location {}'.format('lipdverse_test_'+timestr+'.csv', lipd_test_path[:lipd_test_path.rindex('lipdverse_test')]))
    final_df_test.to_csv(lipd_test_path, sep = ',', encoding = 'utf-8',index = False)

    print('\n Creating file {} at location {}'.format('ground_truth_label_'+timestr+'.json', ground_truth_path[:ground_truth_path.rindex('ground_truth_label')]))
    with open(ground_truth_path, 'w') as ground_truth_label_file:
        json.dump(final_ground_truth_dict, ground_truth_label_file)

if __name__ == '__main__':
    read_latest_data_for_training()
    manually_clean_data_by_replacing_incorrect_values()
    discard_less_frequent_values_from_data()
    downsample_archives_create_final_train_test_data()