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
common_lipdverse_df = pd.read_csv('common_lipdverse_table.csv')

common_lipdverse_df = common_lipdverse_df.replace(np.nan, 'NA', regex=True)
common_lipdverse_df = common_lipdverse_df.replace('Sr Ca', 'Sr/Ca', regex=True)
common_lipdverse_df = common_lipdverse_df.replace('Depth_Cm', 'Depth', regex=True)
common_lipdverse_df = common_lipdverse_df.replace('Depth_M', 'Depth', regex=True)
common_lipdverse_df = common_lipdverse_df.replace('per mille VPDB', 'per mil VPDB', regex=True)
common_lipdverse_df = common_lipdverse_df.replace('per mil VPBD', 'per mil VPDB', regex=True)
common_lipdverse_df = common_lipdverse_df.replace('mmolmol', 'mmol/mol', regex=True)

final_df = common_lipdverse_df.filter(['archiveType','proxyObservationType', 'units', 'interpretation/variable', 'interpretation/variableDetail'], axis=1)

archives_map = {'marine sediment': 'MarineSediment', 'lake sediment': 'LakeSediment', 'glacier ice': 'GlacierIce', 'documents': 'Documents', 'borehole': 'Rock', 'tree': 'Wood', 'bivalve': 'MollusckShell', 'coral': 'Coral', '': '', 'speleothem': 'Speleothem', 'sclerosponge': 'Sclerosponge', 'hybrid': 'Hybrid', 'Sclerosponge': 'Sclerosponge', 'Speleothem': 'Speleothem', 'Coral': 'Coral', 'MarineSediment': 'MarineSediment', 'LakeSediment': 'LakeSediment', 'GlacierIce': 'GlacierIce', 'Documents': 'Documents', 'Hybrid': 'Hybrid', 'MolluskShell': 'MolluskShell', 'Lake': 'Lake'}

for i, row in final_df.iterrows():
    final_df.at[i,'archiveType'] = archives_map[row[0]] if row[0] in archives_map else row[0] 
    
counter_proxy = collections.Counter(final_df['proxyObservationType'])
counter_int_var = collections.Counter(final_df['interpretation/variable'])
counter_int_det = collections.Counter(final_df['interpretation/variableDetail'])

counter_proxy_a = {key:value for key,value in dict(counter_proxy).items() if value > 5}
counter_int_var_a = {key:value for key,value in dict(counter_int_var).items() if value > 1}
counter_int_det_a = {key:value for key,value in dict(counter_int_det).items() if value > 1}

# discard set for proxyObservationType
discard_set = set(counter_proxy.keys()).difference(set(counter_proxy_a.keys()))
final_df = final_df[~final_df['proxyObservationType'].isin(discard_set)]

# discard set for interpretation variable
discard_set = set(counter_int_var.keys()).difference(set(counter_int_var_a.keys()))
final_df = final_df[~final_df['interpretation/variable'].isin(discard_set)]

# discard set for interpretation variable detail
discard_set = set(counter_int_det.keys()).difference(set(counter_int_det_a.keys()))
final_df = final_df[~final_df['interpretation/variableDetail'].isin(discard_set)]

# downsample for archiveType = 'Wood'
df_majority = final_df[final_df.archiveType=='Wood']
df_rest = final_df[final_df.archiveType!='Wood']

df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=300,     # to match minority class
                                 random_state=123)  # reproducibility

final_df_downsampled = pd.concat([df_majority_downsampled, df_rest])

# write back the final training data to create the model.
final_df_downsampled.to_csv('lipdverse_downsampled.csv', sep = ',', encoding = 'utf-8',index = False)