# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:46:47 2021

@author: shrav
"""
import lipd
import pandas as pd
import os
import sys


from sys import platform as _platform

if _platform == "win32":
    sys.path.insert(1, '..\\')
else:
    sys.path.insert(1, '../')

from utils import proxyObsTypeutils
from utils import inferredVarTypeutils


def get_char_string(string):
    '''
    Method to replace a non-ASCII char with a space.

    Parameters
    ----------
    string : str
        Input string.

    Returns
    -------
    n_string : str
        Updated string

    '''
    n_string = ''
    for c in string:
        if 48 <= ord(c) <= 57 or 65 <= ord(c) <= 90 or 97 <= ord(c) <= 122:
            n_string += c
        else:
            n_string += ' '
    return n_string

inf_var_units_map, ignored_inferred_list = None, None

def initialize_data():
    
    global inf_var_units_map, ignored_inferred_list
    
    inf_var_units_map = {'rep diff.': 'NA',
     'Air Surface Temperature': 'degC',
     'Surface D18Osw': 'permil',
     'NA': 'NA',
     'Temperature': 'degC',
     'Sea Surface Temperature': 'degC',
     'Air Condensationlevel Temperature': 'degC',
     'Air Temperature': 'degC',
     'D18O': 'permil',
     'Sedimentation Rate': 'cm/kyr',
     'Thermocline Temperature': 'degC',
     'uncertainty_temperature': 'degC',
     'Sea Surface Salinity': 'psu',
     'Sea Surface D18Osw': 'permil',
     'Lake Surface Temperature': 'degC',
     'Sea Surface Temperature And Salinity': 'NA',
     'Bottom Water D18Osw': 'permil',
     'period': 'NA',
     'D18Ocorr': 'permil',
     'D18Osw': 'permil',
     'Surface Water D18Osw': 'permil',
     'Subsurface Temperature': 'degC',
     'Relative Sea Level': 'm',
     'Subsurface D18Osw': 'permil',
     'Surface Water Temperature': 'degC',
     'Bottom Water Temperature': 'degC',
     'Surface Temperature' : 'degC'
    }
    ignored_inferred_list = ['age', 'depth', 'year', 'Age', 'Depth', 'Year']



def read_lipd_files_list(lipd_files_list):
    '''
    Method to iterate over a list of LiPD files to extract 'paleodata' from it.
    We are currently focusing only on the following attributes from the file;
    archiveType, proxyObservationType, proxyObservationTypeUnits, interpretation/variable, interpretation/variableDetail, inferredVariableType, inferredVariableUnits
    

    Parameters
    ----------
    lipd_files_list : list
        List containing complete path of LiPD files to read.

    Returns
    -------
    table_com : dataframe
        Dataframe consisting of extracted data; filename, compilation, archiveType, proxyObservationType, proxyObservationTypeUnits, interpretation/variable, interpretation/variableDetail, predicted inferredVariableType, predicted inferredVariableUnits
    inf_table_com : dataframe
        Dataframe consisting of extracted data;  filename, compilation, archiveType, inferredVariableType, inferredVariableUnits

    '''
    count = 0
    
    table = pd.DataFrame(columns = ['publication','filename','archiveType', 'variableType', 'proxyObservationType','units', 'rank', 'interpretation/variable','interpretation/variableDetail', 'inferredVariable', 'inferredVarUnits'])
    inf_table = pd.DataFrame(columns = ['publication','filename','archiveType', 'variableType','interpretation/variable','interpretation/variableDetail', 'inferredVariable', 'inferredVarUnits'])

    for line in lipd_files_list:
        print(line)
        count += 1
        print('file count:', count)
        
        full_file_path = line.strip()
        
        file_split = os.path.split(full_file_path)
        filen = file_split[1]
        publication = os.path.split(file_split[0])[1]
        cwd = os.getcwd()
        d = lipd.readLipd(line)
        os.chdir(cwd)
        if 'paleoData' not in d:
            continue
        path = d['paleoData']['paleo0']['measurementTable']['paleo0measurement0']['columns']
        archive = d['archiveType']
        
        for key in path.keys() :
            
            vtype = 'NA'
            unit = 'NotApplicable'
            proxyOType = 'NA'
            intVariable = 'NA'
            intVarDet = 'NA'
            rank = 'NA'
            infVar = 'NA'
            infVarUnits = 'NA'
            if 'variableType' in path[key].keys():
                vtype = path[key]['variableType']
                
            if vtype == 'measured':
                if 'proxyObservationType' in path[key].keys() :
                    proxyOType = path[key]['proxyObservationType']
                if proxyOType == 'NA' and 'variableName' in path[key].keys():
                    vname = path[key]['variableName']
                    proxyOType, rem = proxyObsTypeutils.predict_proxy_obs_type_from_variable_name(vname)
                    if proxyOType == 'NA':
                        proxyOType = vname
                
                if type(proxyOType) == str and not proxyOType.isupper():
                    proxyOType = proxyOType.title()

                if 'units' in path[key].keys() :
                        unit = path[key]['units']
                
                if len(proxyOType) > 45:
                    proxyOType = 'NA'
                    unit = 'NA'
                        
                inter_set = False
                if 'interpretation' in path[key].keys() :
                    inter_len = len(path[key]['interpretation'])
                    for inter in path[key]['interpretation']:
                        infVar = 'NA'
                        infVarUnits = 'NA'
                        if type(inter) is not str :
                            if 'variable' in inter.keys() :
                                intVariable = inter['variable']
                                intVariable = intVariable.title()
                                if intVariable == 'T':
                                    intVariable = 'Temperature'
                                elif intVariable == 'P':
                                    intVariable == 'Pressure'
                            if 'variableDetail' in inter.keys() :
                                intVarDet = get_char_string(inter['variableDetail'])
                                intVarDet = intVarDet.title()
                            if 'rank' in inter.keys() :
                                rank = inter['rank']
                            else:
                                rank = inter_len
                            
                            if len(intVariable) > 45:
                                intVariable = 'NA'
                            if len(intVarDet) > 45:
                                intVarDet = 'NA'
                            
                            if infVar == 'NA' and intVariable != 'NA' and intVarDet != 'NA':
                                inf_from_interp = (' ').join([intVarDet, intVariable])
                                infVar = inf_from_interp
                                infVarUnits = inf_var_units_map[infVar] if infVar in inf_var_units_map else 'NA'
                                    
    
                            if unit != 'NotApplicable' or proxyOType != 'NA':
                                df = pd.DataFrame({'publication':[publication],'filename':[filen], 'archiveType': [archive],'variableType':[vtype], 'units':[unit],'proxyObservationType':[proxyOType],'rank':[rank],'interpretation/variable':[intVariable],'interpretation/variableDetail':[intVarDet], 'inferredVariable':[infVar], 'inferredVarUnits':[infVarUnits]})
                                table = table.append(df, ignore_index = True)
                                inter_set = True
                                
                if not inter_set and (unit != 'NotApplicable' or proxyOType != 'NA'):
                    df = pd.DataFrame({'publication':[publication],'filename':[filen], 'archiveType': [archive],'variableType':[vtype], 'units':[unit],'proxyObservationType':[proxyOType],'rank':[rank],'interpretation/variable':[intVariable],'interpretation/variableDetail':[intVarDet], 'inferredVariable':[infVar], 'inferredVarUnits':[infVarUnits]})
                    table = table.append(df, ignore_index = True) 
                                
            elif vtype == 'inferred':
                if 'inferredVariableType' in path[key].keys() :
                    infVar = path[key]['inferredVariableType']
                    if infVar in ignored_inferred_list:
                        continue
                if infVar == 'NA' and 'variableName' in path[key].keys() :
                    vname = path[key]['variableName']
                    infVar, rem = inferredVarTypeutils.predict_inf_var_type_from_variable_name(vname)
                    if infVar == 'NA':
                        infVar = vname
                    elif len(infVar) > 45:
                        infVar = 'NA'
                if 'units' in path[key].keys() :
                        infVarUnits = path[key]['units']
                
                inter_set = False
                if 'interpretation' in path[key].keys() :
                    inter_len = len(path[key]['interpretation'])
                    for inter in path[key]['interpretation']:
                        if type(inter) is not str :
                            if 'variable' in inter.keys() :
                                intVariable = inter['variable']
                                intVariable = intVariable.title()
                                if intVariable == 'T':
                                    intVariable = 'Temperature'
                                elif intVariable == 'P':
                                    intVariable == 'Pressure'
                            if 'variableDetail' in inter.keys() :
                                intVarDet = get_char_string(inter['variableDetail'])
                                intVarDet = intVarDet.title()
                            
                            if len(intVariable) > 45:
                                intVariable = 'NA'
                            if len(intVarDet) > 45:
                                intVarDet = 'NA'

                            if 'rank' in inter.keys() :
                                rank = inter['rank']
                            else:
                                rank = inter_len
                
                if infVar != 'NA':
                    df = pd.DataFrame({'publication':[publication],'filename':[filen], 'archiveType': [archive],'variableType':[vtype], 'rank':[rank],'interpretation/variable':[intVariable],'interpretation/variableDetail':[intVarDet], 'inferredVariable':[infVar], 'inferredVarUnits':[infVarUnits]})
                    inf_table = inf_table.append(df, ignore_index = True) 
    
    table_com=table.explode('proxyObservationType').explode('units').explode('rank').explode('variableType').explode('interpretation/variable').explode('interpretation/variableDetail').explode('inferredVariable').explode('inferredVarUnits').reset_index()
    table_com = table_com.drop(columns = ['index'])
    inf_table_com=inf_table.explode('rank').explode('variableType').explode('interpretation/variable').explode('interpretation/variableDetail').explode('inferredVariable').explode('inferredVarUnits').reset_index()
    inf_table_com = inf_table_com.drop(columns = ['index'])
    
    return table_com, inf_table_com

if __name__ == '__main__':
    initialize_data()
    print(proxyObsTypeutils.proxy_obs_map)
else:
    initialize_data()