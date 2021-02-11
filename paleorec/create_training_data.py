# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 15:39:08 2021

@author: shrav
"""

import pandas as pd
import numpy as np
import collections
from sklearn.utils import resample

'''
 File to create class - balanced training data from wiki.linkedearth lipd files.

'''
common_lipdverse_df = pd.read_csv('..\merged_common_lipdverse_inferred.csv')


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

final_df = common_lipdverse_df.filter(['archiveType','proxyObservationType', 'units', 'interpretation/variable', 'interpretation/variableDetail', 'inferredVariable', 'inferredVarUnits'], axis=1)

archives_map = {'marine sediment': 'MarineSediment', 'lake sediment': 'LakeSediment', 'glacier ice': 'GlacierIce', 'documents': 'Documents', 'borehole': 'Rock', 'tree': 'Wood', 'bivalve': 'MollusckShell', 'coral': 'Coral', '': '', 'speleothem': 'Speleothem', 'sclerosponge': 'Sclerosponge', 'hybrid': 'Hybrid', 'Sclerosponge': 'Sclerosponge', 'Speleothem': 'Speleothem', 'Coral': 'Coral', 'MarineSediment': 'MarineSediment', 'LakeSediment': 'LakeSediment', 'GlacierIce': 'GlacierIce', 'Documents': 'Documents', 'Hybrid': 'Hybrid', 'MolluskShell': 'MolluskShell', 'Lake': 'Lake'}

for i, row in final_df.iterrows():
    final_df.at[i,'archiveType'] = archives_map[row[0]] if row[0] in archives_map else row[0] 

final_df = final_df[final_df.units != 'Mg/Ca']
    
counter_arch = collections.Counter(final_df['archiveType'])
counter_proxy = collections.Counter(final_df['proxyObservationType'])
counter_units = collections.Counter(final_df['units'])
counter_int_var = collections.Counter(final_df['interpretation/variable'])
counter_int_det = collections.Counter(final_df['interpretation/variableDetail'])
counter_inf_var = collections.Counter(final_df['inferredVariable'])
counter_inf_var_units = collections.Counter(final_df['inferredVarUnits'])

counter_proxy_a = {key:value for key,value in dict(counter_proxy).items() if value > 5}
counter_int_var_a = {key:value for key,value in dict(counter_int_var).items() if value > 1}
counter_int_det_a = {key:value for key,value in dict(counter_int_det).items() if value > 1}

# MANAUL TASK - SCAN THROUGH ALL THE COUNTER VARIABLES TO CHECK IF WE NEED TO OMIT ANY-CO-k( WHERE K CAN BE 1-5, DEPENDING ON THE USE-CASE)

# discard set for proxyObservationType
discard_set = set(counter_proxy.keys()).difference(set(counter_proxy_a.keys()))
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

# MANUAL TASK - DECIDE WHICH ARCHIVES NEED TO BE DOWN-SAMPLED
# CURRENTLY WE ARE ONLY DOING WOOD, BECAUSE WE HAVE AROUND 2000 SAMPLES.

# downsample for archiveType = 'Wood'
df_wood = final_df[final_df.archiveType=='Wood']
df_marine_sed = final_df[final_df.archiveType=='MarineSediment']
discard_set = {'Wood', 'MarineSediment'}
df_rest = final_df[~final_df['archiveType'].isin(discard_set)]

df_wood_downsampled = resample(df_wood, 
                                 replace=False,    # sample without replacement
                                 n_samples=350,     # to match minority class
                                 random_state=123)  # reproducibility

df_marine_downsampled = resample(df_marine_sed, 
                                 replace=False,    # sample without replacement
                                 n_samples=350,     # to match minority class
                                 random_state=123)  # reproducibility

# MANUAL TASK - ADD DATA FOR WOOD FROM INFERRED VARIABLE TYPE CSV FILE,
# BECAUSE THERE ARE NO SAMPLES WITH UNITS FOR INFERRED VARIABLE TYPE AND INFERRED VARIABLE TYPE UNITS FOR ARCHIVE = WOOD

wood_inferred_df = pd.read_csv('..\wood_inferred_data.csv')
wood_inferred_df = wood_inferred_df.replace(np.nan, 'NA', regex=True)
final_df_downsampled = pd.concat([df_wood_downsampled, wood_inferred_df, df_marine_downsampled, df_rest])

final_df_downsampled = final_df_downsampled.sample(frac=1, random_state=2021).reset_index(drop=True)

# write back the final training data to create the model.
final_df_downsampled.to_csv('..\lipdverse_downsampled.csv', sep = ',', encoding = 'utf-8',index = False)



