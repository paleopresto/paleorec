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

if _platform == "win32":
    sys.path.insert(1, '..\\')
else:
    sys.path.insert(1, '../')
from utils import fileutils

'''
 File to create class - balanced training data from wiki.linkedearth lipd files.

'''

common_lipdverse_df, final_df = None, None
names_set_dict = {}
counter_arch = {}


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
    common_lipdverse_df = common_lipdverse_df.replace('Surface P','Surface Pressure', regex=True)
    common_lipdverse_df = common_lipdverse_df.replace('sed rate','Sedimentation Rate', regex=True)
    common_lipdverse_df = common_lipdverse_df.replace('d18ocorr','D18Ocorr', regex=True)

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

def discard_less_frequent_values_from_data():
    '''
    This method reduces the subset of data to the fields in the chain, 
    i.e archiveType, proxyObservationType, units, interpretation/variable, interpretation/variableDetail, inferredVariable, inferredVarUnits.
    
    Create a dict to store autocomplete information for each fieldType.
    
    Generate a counter for the values in each column to understand the distribution of each individual field.
    Manually decide whether to eliminate any co 1 values within each field.
    Uncomment the code to print each of the counter fields to make the decision.
    
    Update the dataframe by discarding those values from there as well.

    Returns
    -------
    None.

    '''
    
    global final_df, names_set_dict, counter_arch
    
    
    final_df = common_lipdverse_df.filter(['archiveType','proxyObservationType', 'units', 'interpretation/variable', 'interpretation/variableDetail', 'inferredVariable', 'inferredVarUnits'], axis=1)
    
    archives_map = {'marine sediment': 'MarineSediment', 'lake sediment': 'LakeSediment', 'glacier ice': 'GlacierIce', 'documents': 'Documents', 'borehole': 'Rock', 'tree': 'Wood', 'bivalve': 'MollusckShell', 'coral': 'Coral', '': '', 'speleothem': 'Speleothem', 'sclerosponge': 'Sclerosponge', 'hybrid': 'Hybrid', 'Sclerosponge': 'Sclerosponge', 'Speleothem': 'Speleothem', 'Coral': 'Coral', 'MarineSediment': 'MarineSediment', 'LakeSediment': 'LakeSediment', 'GlacierIce': 'GlacierIce', 'Documents': 'Documents', 'Hybrid': 'Hybrid', 'MolluskShell': 'MolluskShell', 'Lake': 'Lake', 'molluskshell': 'MollusckShell', 'Wood': 'Wood', 'Rock': 'Rock'}
    
    for i, row in final_df.iterrows():
        final_df.at[i,'archiveType'] = archives_map[row[0]] if row[0] in archives_map else row[0] 
    
    final_df = final_df[final_df.units != 'Mg/Ca']
    
    
    counter_arch = collections.Counter(final_df['archiveType'])
    # print('ARCHIVE TYPES : ', counter_arch)
    
    counter_proxy = collections.Counter(final_df['proxyObservationType'])
    # print('PROXY OBSERVATION TYPES : ',counter_proxy)
    
    counter_units = collections.Counter(final_df['units'])
    # print('PROXY OBSERVATION TYPE UNITS : ',counter_units)
    
    counter_int_var = collections.Counter(final_df['interpretation/variable'])
    # print('INTERPRETATION/VARIABLE : ',counter_int_var)
    
    counter_int_det = collections.Counter(final_df['interpretation/variableDetail'])
    # print('INTERPRETATION/VARIABLE DETAIL : ',counter_int_det)
    
    counter_inf_var = collections.Counter(final_df['inferredVariable'])
    # print('INFERRED VARIABLE : ', counter_inf_var)
    
    counter_inf_var_units = collections.Counter(final_df['inferredVarUnits'])
    # print('INFERRED VARIABLE UNITS : ',counter_inf_var_units)
    
    # Add to a file for autocomplete suggestions without removing any co 1
    names_set_dict = {'proxyObservationType' : list(counter_proxy.keys()), 'proxyObservationTypeUnits' : list(counter_units.keys()), 
                      'interpretation/variable' : list(counter_int_var.keys()), 'interpretation/variableDetail' : list(counter_int_det.keys()), 
                      'inferredVariable' : list(counter_inf_var.keys()), 'inferredVariableUnits' : list(counter_inf_var_units.keys())}
    write_autocomplete_data_file()
    
    
    counter_proxy_a = {key:value for key,value in dict(counter_proxy).items() if value > 5}
    counter_int_var_a = {key:value for key,value in dict(counter_int_var).items() if value > 1}
    counter_int_det_a = {key:value for key,value in dict(counter_int_det).items() if value > 1}
    
    # MANAUL TASK - SCAN THROUGH ALL THE COUNTER VARIABLES TO CHECK IF WE NEED TO OMIT ANY-CO-k( WHERE K CAN BE 1-5, DEPENDING ON THE USE-CASE)
    
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
    discard_set = {'Calendar Age', 'Accumulation rate', 'k37 n toc', 'acc rate k37', 'acc rate toc', 'Age', 'Radiocarbon Age', 'mg/ca'}
    final_df = final_df[~final_df['inferredVariable'].isin(discard_set)]
    
    
    # DROP NA VALUES FROM THE ARCHIVE TYPE FIELD AS IT CAN NEVER CONTAIN NA
    final_df = final_df[final_df.archiveType != 'NA']
    final_df.dropna(subset=['archiveType'], inplace=True)

def downsample_archives_create_final_train_test_data():
    '''
    
    Manually decide based on the counter for archiveTypes which archiveTypes need to be downsampled.
    Currently we are downsampling Wood and Marine Sediment to include 350 samples of each.
    We are including all samples for all the other archiveTypes.
    
    Simulataneously creating a test dataset by resampling from the training data.
    Since we do not even distribution of data across each class, we have used 'stratify' during resample.
    This will help us even out the distribution of data across all classess in the provided dataset.
    
    
    Returns
    -------
    None.

    '''

    # MANUAL TASK - DECIDE WHICH ARCHIVES NEED TO BE DOWN-SAMPLED
    # CURRENTLY WE ARE ONLY DOING WOOD, BECAUSE WE HAVE AROUND 2000 SAMPLES.
    
    print('ARCHIVE TYPES TO DOWNSAMPLE: ', counter_arch)
    
    # downsample for archiveType = 'Wood' and 'MarineSediment'
    df_wood = final_df[final_df.archiveType=='Wood']
    df_marine_sed = final_df[final_df.archiveType=='MarineSediment']
    discard_set = {'Wood', 'MarineSediment'}
    df_rest = final_df[~final_df['archiveType'].isin(discard_set)]
    
    df_wood_downsampled = resample(df_wood, 
                                     replace=False,    # sample without replacement
                                     n_samples=350,     # to match minority class
                                     random_state=27,  # reproducibility
                                     stratify=df_wood)
    
    df_marine_downsampled = resample(df_marine_sed, 
                                     replace=False,    # sample without replacement
                                     n_samples=350,     # to match minority class
                                     random_state=100,  # reproducibility
                                     stratify=df_marine_sed)
    
    # Add all unique values from the df_wood and df_marine_sed data frame into the downsampled dataframes.
    # We intend to provide our model all the unique values that are currently present in the data
    df_wood_nodup = df_wood.drop_duplicates()
    df_wood_ds_no_dup = df_wood_downsampled.drop_duplicates()
    
    df_wood_extra = df_wood_nodup.merge(df_wood_ds_no_dup, how='left', indicator=True)
    df_wood_extra = df_wood_extra[df_wood_extra['_merge']=='left_only']
    df_wood_extra = df_wood_extra.drop(columns=['_merge'])
    
    df_wood_downsampled = df_wood_downsampled.append(df_wood_extra, ignore_index=True)
    
    df_ms_nodup = df_marine_sed.drop_duplicates()
    df_ms_ds_no_dup = df_marine_downsampled.drop_duplicates()
    
    df_ms_extra = df_ms_nodup.merge(df_ms_ds_no_dup, how='left', indicator=True)
    df_ms_extra = df_ms_extra[df_ms_extra['_merge']=='left_only']
    df_ms_extra = df_ms_extra.drop(columns=['_merge'])
    
    df_marine_downsampled = df_marine_downsampled.append(df_ms_extra, ignore_index=True)
    
    df_wood_test = resample(df_wood_downsampled, 
                            replace=False,    # sample without replacement
                            n_samples=10,     # to match minority class
                            random_state=123,  # reproducibility
                            stratify=df_wood_downsampled)
    
    
    df_marine_test = resample(df_marine_downsampled, 
                                replace=False,    # sample without replacement
                                n_samples=12,     # to match minority class
                                random_state=123,  # reproducibility 
                                stratify=df_marine_downsampled)
     
    # MANUAL TASK - ADD DATA FOR WOOD FROM INFERRED VARIABLE TYPE CSV FILE,
    # BECAUSE THERE ARE NO SAMPLES WITH UNITS FOR INFERRED VARIABLE TYPE AND INFERRED VARIABLE TYPE UNITS FOR ARCHIVE = WOOD
    
    
    if _platform == "win32":
        wood_inferred_path = '..\data\csv\wood_inferred_data.csv'
    else:
        wood_inferred_path = '../data/csv/wood_inferred_data.csv'
    
    wood_inferred_df = pd.read_csv(wood_inferred_path)
    wood_inferred_df = wood_inferred_df.replace(np.nan, 'NA', regex=True)
    
    wood_inferred_test = resample(wood_inferred_df, 
                                     replace=False,    # sample without replacement
                                     n_samples=2,     # to match minority class
                                     random_state=123)  # reproducibility
    
    df_rest_test = resample(df_rest, 
                            replace=False,    # sample without replacement
                            n_samples=26,     # to match minority class
                            random_state=123,  # reproducibility
                            stratify=df_rest)
    
    final_df_downsampled = pd.concat([df_wood_downsampled, wood_inferred_df, df_marine_downsampled, df_rest])
    final_df_test = pd.concat([df_wood_test, df_marine_test, wood_inferred_test, df_rest_test])
    
    final_df_downsampled = final_df_downsampled.sample(frac=1, random_state=2021).reset_index(drop=True)
    final_df_test = final_df_test.sample(frac=1, random_state=2021).reset_index(drop=True)
    
    
    
    timestr = time.strftime("%Y%m%d_%H%M%S")
    
    
    if _platform == "win32":
        lipd_downsampled_path = '..\data\csv\lipdverse_downsampled_'+timestr+'.csv'
        lipd_test_path = '..\data\csv\lipdverse_test_'+timestr+'.csv'
        autocomplete_file_path = '..\\data\\autocomplete\\autocomplete_file_'+timestr+'.json'
    else:
        lipd_downsampled_path = '../data/csv/lipdverse_downsampled_'+timestr+'.csv'
        lipd_test_path = '../data/csv/lipdverse_test_'+timestr+'.csv'
        autocomplete_file_path = '../data/autocomplete/autocomplete_file_'+timestr+'.json'
    
    # write back the final training data to create the model.
    final_df_downsampled.to_csv(lipd_downsampled_path, sep = ',', encoding = 'utf-8',index = False)
    # write back the final test data to calculate accuracy of the model.
    final_df_test.to_csv(lipd_test_path, sep = ',', encoding = 'utf-8',index = False)
    
    with open(autocomplete_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(names_set_dict, json_file)

