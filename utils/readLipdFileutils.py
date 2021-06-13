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

inf_var_units_map, ignore_list = None, None
interp_map = {}
inf_map = {}
interp_det_discard,interp_ignore_set = set(), set()

def initialize_data():
    
    global inf_var_units_map, ignore_list, interp_map, interp_det_discard,interp_ignore_set, inf_map
    
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
    ignore_list = ['age', 'depth', 'year', 'Age', 'Depth', 'Year', 'Not Clear', 'No Isotopic Analyes Here', 'nonReliableTemperature 1', 'sampleID']

    interp_map['P_Isotope'] = 'Precipitation Isotope'
    interp_map['T_Water'] = 'Water Temperature'
    interp_map['D18O_Seawater'] = 'D18O Seawater'
    interp_map['Temperature_Water'] = 'Water Temperature'
    interp_map['Rainfall And Temperature'] = 'Temperature and Precipitation'
    interp_map['P_E'] = 'Precipitation and Evaporation'
    interp_map['Nan'] = 'NA'
    interp_map['E/P'] = 'Precipitation and Evaporation'
    interp_map['D18Op'] = 'D18O Of Precipitation'
    interp_map['P/E'] = 'Precipitation and Evaporation'
    interp_map['?18O.Precipitation'] = 'D18O Of Precipitation'
    interp_map['Pdo'] = 'Pacific Decadal Oscillation'
    interp_map['Rainfall Source Area'] = 'Source Region'
    interp_map['P_Amount'] = 'Precipitation Amount'
    interp_map['Moisture Balance (P-E)'] = 'Precipitation and Evaporation'
    interp_map['Surface P'] = 'Surface Pressure'
    interp_map['E:P (Groundwater "Fluid Balance")'] = 'Precipitation and Evaporation'
    interp_map['T_Air'] = 'Air Temperature'
    interp_map['Nao'] = 'NAO'
    interp_map['D18O Precipitation'] = 'D18O Of Precipitation'
    interp_map['T_Lake'] = 'Lake Temperature'
    interp_map['Evaporation/Precipitation'] = 'Precipitation and Evaporation'
    interp_map['Evaporation/ Precipitation'] = 'Precipitation and Evaporation'
    interp_map['Rh'] = 'Relative Humidity'
    interp_map['Temperature and Precipitation Amount'] = 'Temperature and Precipitation'
    interp_map['Evaptranspiration'] = 'Evapotranspiration'
    interp_map['Moisture Source'] = 'Sea Ice'
    interp_map['Temperature/D18Osw'] = 'Temperature and D18O of Seawater'
    interp_map['Seawater_Isotope'] = 'Seawater Isotope'
    interp_map['Carbonate_Ion_Concentration'] = 'Carbonate Ion Concentration'
    interp_map['Regional Rainfall Amount'] = 'Precipitation Amount'
    interp_map['Temperature/Salinity'] = 'Temperature And Salinity'
    interp_map['Amo'] = 'Atlantic Multi-decadal Oscillation'
    interp_map['Rainfall Seasonality'] = 'Precipitation Seasonality'
    interp_map['Temperature_Air'] = 'Air Temperature'
    interp_map['Precipitation_Amount Temperature_Air'] = 'Temperature and Precipitation'
    interp_map['Precipitation_Amount Sam'] = 'Precipitation Amount'
    interp_map['Precipitation_Amount Humidity'] = 'Precipitation amount and humidity'
    interp_map['Salinity_Seawater'] = 'Salinity'
    interp_map['P_Amount And Temperature'] = 'Temperature and Precipitation'
    interp_map['T_Air Rh P_Amount'] = 'Precipitation amount and humidity'
    interp_map['P_Amount T_Air'] = 'Temperature and Precipitation'
    interp_map['P_Amount P_E'] = 'Precipitation and Evaporation'
    interp_map['Precipitation D18O'] = 'D18O Of Precipitation'
    interp_map['P_Amount Rh'] = 'Precipitation amount and humidity'
    interp_map['P_Amount Rh T_Air P_E'] = 'Precipitation and evaporation; temperature'
    interp_map['Pdsi'] = 'Palmer Drought Severity Index'
    interp_map['T_Air P_Amount Drought Index Spei'] = 'SPEI'
    interp_map['T_Air P_Amount'] = 'Temperature and Precipitation'
    interp_map['Et'] = 'Evapotranspiration'
    interp_map['Relative Humdity'] = 'Relative Humidity'

    interp_det_discard = {'Na', 'Air', 'Of Precipitation', 'Tropical Or North Pacific Moisture', 'Precipitation', 'Moisture Source', 'Indian Monsoon Strength', 'In The Venezuelan Andes', 'South China Sea', 'In The Southern Tropical Andes','Continental Sweden', 'Strength   Position Of Aleutian Low', 'Rain', 'D18Op', 'Southern Tibet', 'Qaidam Basin', 'Aleutian Low Westerly Storm Trajectories', 'Moisture', 'Enso Pdo', 'Enso  Pacific Ocean Atmosphere Dynamics ', 'E P Lake Water', 'Summer Monsoon', 'Central Asia', 'Central India', 'At  39 Degrees Lat', 'Coastal Lagoon Water', 'Lake  Winds In Eastern Patagonia', 'Relative Amount Of Winter Snowfall', 'Annual Amount Weighted Precipitation Isotopes', 'Evaporative Enrichment Of Leaf Water', '0 58    0 11Ppt Degrees C', 'Surface Waters Of Polar Origin', 'Amount Of Rainfall Change', 'East Asian Monsoon Strength', 'Variations In Winter Temperature In The Alps', 'Sam', 'Regional And Hemispheric Temperature', 'Monsoon Strength', 'Summer Monsoon Rainfall Amount', 'Australian Indonesian Monsoon Rainfall', '20 Mbsl', 'Seasonal  Annual', 'Seasonal', 'Soil Moisture Stress', 'D18O Of The Leaf Water', 'Soil Moisture', 'Precipitation D18O', 'Modelled Precipitation D18O', 'Leaf Water D18O', 'Maximum Air Temperature  Seasonal', 'Maximum Temperature', 'Relative Humidity', 'Atmospheric Surface Temperature', 'Soil Pdsi Conditions', 'Isotopic Composition Of Summer Rainfall', 'Souther Annular Mode  Sam ', 'Net Primary Productivity'}

    interp_ignore_set = {'Seasonal', 'Annual', 'North', 'South', 'East', 'West', 'Northern', 'Southern', 'Eastern', 'Western', 'Tropical','China', 'India', 'Aleutian', 'Asia', 'Alps', 'Summer', 'Winter', 'Polar', 'Monsoon', 'Central'}

    inf_map = {'deep.temp': 'Deep Water Temperature', 'soilTemp' : 'Soil Temperature', 'temperatureComposite' : 'Temperature', 'temperature1' : 'Temperature', 'tempSource' : 'Temperature at the source', 'Lake Water Water Temperature' : 'Lake Water Temperature', 'Lake Water D18O Of Precipitation' : 'Precipitation D18O', 'Surface Relative Humidity': 'Relative Humidity', 'Air Surface Precipitation Amount':'Precipitation Amount'}

def validate_inf_after_appending(inf_var):

    if inf_var in ignore_list or 'age' in inf_var.lower() or 'error' in inf_var.lower() or 'uncertainty' in inf_var.lower() or 'sampleid' in inf_var.lower():
        return 'NA'
    elif 'Precipitation Isotope' in inf_var:
        return 'Precipitation Isotope'
    elif 'Precipitation And Evaporation' in inf_var:
        return 'Precipitation And Evaporation'
    elif 'Pacific Decadal Oscillation' in inf_var:
        return 'Pacific Decadal Oscillation'
    elif 'DBD' == inf_var:
        return 'Dry Bulk Density'
    elif inf_var in inf_map:
        return inf_map[inf_var]

    inf_var = inf_var.replace('Water', '')

    if 'sub' in inf_var.lower():
        new_inf_var = []
        for word in inf_var.split(' '):
            if word != 'D18O' and any(map(str.isdigit, inf_var)):
                continue
            new_inf_var.append(word)

        new_inf_var = ' '.join(new_inf_var)
        return new_inf_var
    elif 'mixed' in inf_var.lower():
        return inf_var
    elif inf_var.startswith('Surface') and 'sea' not in inf_var.lower():
        inf_var = inf_var.split(' ')
        inf_var.insert(0, 'Sea')
        inf_var = ' '.join(inf_var)
        return inf_var

    return inf_var

def finalize_data(dataframe):

    dataframe = dataframe.replace('Nan', 'NA', regex=True)
    dataframe = dataframe.replace('g.cm-2.a-1', 'g/cm2a', regex=True)
    dataframe = dataframe.replace('mcm', 'microm', regex=True)
    dataframe = dataframe.replace('NotApplicable', 'NA', regex=True)
    dataframe = dataframe.replace('Sub Surface', 'Subsurface', regex=True)

    return dataframe

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
    
    table = pd.DataFrame(columns = ['coordinates','publication','filename','archiveType', 'variableType','description', 'proxyObservationType','units', 'rank', 'interpretation/variable','interpretation/variableDetail', 'inferredVariable', 'inferredVarUnits'])
    inf_table = pd.DataFrame(columns = ['coordinates','publication','filename','archiveType', 'variableType','description','interpretation/variable','interpretation/variableDetail', 'inferredVariable', 'inferredVarUnits'])

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

        if 'geo' in d and 'geometry' in d['geo'] and 'coordinates' in d['geo']['geometry']:
            geo_coord = d['geo']['geometry']['coordinates']

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
            des = 'NA'
            if 'variableType' in path[key].keys():
                vtype = path[key]['variableType']
                
            if vtype == 'measured':
                if 'proxyObservationType' in path[key].keys() :
                    proxyOType = path[key]['proxyObservationType']
                    if proxyOType in ignore_list:
                        continue
                if proxyOType == 'NA' and 'variableName' in path[key].keys():
                    vname = path[key]['variableName']
                    proxyOType, rem = proxyObsTypeutils.predict_proxy_obs_type_from_variable_name(vname)
                    if proxyOType == 'NA':
                        proxyOType = vname
              
                proxyOType = proxyObsTypeutils.validate_proxyObsType(proxyOType)
                if type(proxyOType) == str and not proxyOType.isupper():
                    proxyOType = proxyOType.title()

                if 'units' in path[key].keys() :
                        unit = path[key]['units']
                        if '_' in unit:
                            unit = unit.replace('_','/')
                        elif unit == 'PC' or unit == 'zscore':
                            unit = 'NA'
                
                if len(proxyOType) > 45:
                    proxyOType = 'NA'
                    unit = 'NA'
                        
                if 'description' in path[key].keys():
                    des = path[key]['description']

                inter_set = False
                if 'interpretation' in path[key].keys() :
                    inter_len = len(path[key]['interpretation'])
                    for inter in path[key]['interpretation']:
                        infVar = 'NA'
                        infVarUnits = 'NA'
                        if type(inter) is not str :
                            if 'variable' in inter.keys() :
                                intVariable = inter['variable']
                                
                                if intVariable == 'T':
                                    intVariable = 'Temperature'
                                elif intVariable == 'P':
                                    intVariable = 'Pressure'
                            if 'variableDetail' in inter.keys() :
                                intVarDet = get_char_string(inter['variableDetail'])
                                
                            if 'rank' in inter.keys() :
                                rank = inter['rank']
                            else:
                                rank = inter_len
                            
                            if len(intVariable) > 45:
                                intVariable = 'NA'
                            if len(intVarDet) > 45:
                                intVarDet = 'NA'
                            
                            accept = False
                            if intVariable in interp_map:
                                intVariable = interp_map[intVariable]
                            elif intVariable.title() in interp_map:
                                intVariable = interp_map[intVariable.title()]
                            else:
                                for name in intVariable.split(' '):
                                    if name in interp_ignore_set:
                                        accept = True
                                        intVariable = 'NA'
                                        break
                                if accept:
                                    pass
                                else:
                                    intVariable = 'NA'
                            
                            intVariable = intVariable.title() if intVariable != 'NA' and not intVariable.isupper() else intVariable
                            intVarDet = intVarDet.title() if intVarDet != 'NA' else intVarDet
                            if intVariable.lower() in ['pdo', 'amo', 'pdsi']:
                                if intVariable.lower() == 'pdsi':
                                    intVariable = 'Palmer Drought Severity Index'
                                elif intVariable.lower() == 'pdo':
                                    intVariable = 'Pacific Decadal Oscillation'
                                elif intVariable.lower() == 'amo':
                                    intVariable = 'Atlantic Multi-decadal Oscillation'
                                intVariable = intVariable.upper()

                            if intVarDet in interp_det_discard or intVarDet == 'Nan' or intVarDet == 'Na':
                                intVarDet = 'NA'
                            elif intVarDet == 'Surface Temperature' or intVarDet == 'Surface Relative Humidity':
                                intVarDet = 'Surface'
                            else:
                                for name in intVarDet:
                                    if name in interp_ignore_set:
                                        intVarDet = 'NA'
                                        break                            
                            
                            if infVar == 'NA' and intVariable != 'NA' and intVarDet != 'NA' and intVarDet != 'Na':
                                inf_from_interp = (' ').join([intVarDet, intVariable])
                                infVar = inf_from_interp if not inf_from_interp.startswith('Na') else ''
                                # infVar = inf_from_interp
                                infVarUnits = inf_var_units_map[infVar] if infVar in inf_var_units_map else 'NA'
                            
                            if infVar != 'NA':
                                if infVar == 'surface.temp' and archive.lower() in ['marinesediment', 'marine sediment']:
                                    infVar = 'Sea Surface Temperature'
                                else:
                                    infVar = validate_inf_after_appending(infVar)
    
                            if unit != 'NotApplicable' or proxyOType != 'NA':
                                df = pd.DataFrame({'coordinates':[geo_coord],'publication':[publication],'filename':[filen], 'archiveType': [archive],'variableType':[vtype], 'units':[unit],'description':[des],'proxyObservationType':[proxyOType],'rank':[rank],'interpretation/variable':[intVariable],'interpretation/variableDetail':[intVarDet], 'inferredVariable':[infVar], 'inferredVarUnits':[infVarUnits]})
                                table = table.append(df, ignore_index = True)
                                inter_set = True
                                
                if not inter_set and (unit != 'NotApplicable' or proxyOType != 'NA'):
                    df = pd.DataFrame({'coordinates':[geo_coord],'publication':[publication],'filename':[filen], 'archiveType': [archive],'variableType':[vtype], 'units':[unit],'description':[des],'proxyObservationType':[proxyOType],'rank':[rank],'interpretation/variable':[intVariable],'interpretation/variableDetail':[intVarDet], 'inferredVariable':[infVar], 'inferredVarUnits':[infVarUnits]})
                    table = table.append(df, ignore_index = True) 
                                
            elif vtype == 'inferred':
                if 'inferredVariableType' in path[key].keys() :
                    infVar = path[key]['inferredVariableType']
                    if infVar in ignore_list or 'age' in infVar.lower() or 'error' in infVar.lower() or 'uncertainty' in infVar.lower() or 'sampleid' in infVar.lower():
                        infVar = 'NA'
                        continue
                    elif 'nonreliable' in infVar.lower():
                        infVar = 'NA'
                        continue
                    elif 'sampleID' == infVar:
                        infVar = 'NA'
                        continue
                if infVar == 'NA' and 'variableName' in path[key].keys() :
                    vname = path[key]['variableName']
                    infVar, rem = inferredVarTypeutils.predict_inf_var_type_from_variable_name(vname)
                    if infVar == 'NA' and vname not in ignore_list and 'age' not in vname.lower() and 'year' not in vname.lower() and 'sampleid' in vname.lower():
                        infVar = vname
                    elif len(infVar) > 45:
                        infVar = 'NA'
                
                infVar = validate_inf_after_appending(infVar)
                if 'units' in path[key].keys() :
                        infVarUnits = path[key]['units']
                        if '_' in infVarUnits:
                            infVarUnits = infVarUnits.replace('_','/')
                        elif unit == 'PC' or unit == 'zscore':
                            unit = 'NA'
                
                if 'description' in path[key].keys():
                    des = path[key]['description']

                inter_set = False
                if 'interpretation' in path[key].keys() :
                    inter_len = len(path[key]['interpretation'])
                    for inter in path[key]['interpretation']:
                        if type(inter) is not str :
                            if 'variable' in inter.keys() :
                                intVariable = inter['variable']
                                if intVariable == 'T':
                                    intVariable = 'Temperature'
                                elif intVariable == 'P':
                                    intVariable = 'Pressure'
                            if 'variableDetail' in inter.keys() :
                                intVarDet = get_char_string(inter['variableDetail'])
                                
                            
                            if len(intVariable) > 45:
                                intVariable = 'NA'
                            if len(intVarDet) > 45:
                                intVarDet = 'NA'
                            accept = False
                            if intVariable in interp_map:
                                intVariable = interp_map[intVariable]
                            elif intVariable.title() in interp_map:
                                intVariable = interp_map[intVariable.title()]
                            else:
                                for name in intVariable.split(' '):
                                    if name in interp_ignore_set:
                                        accept = True
                                        intVariable = 'NA'
                                        break
                                if accept:
                                    pass
                                else:
                                    intVariable = 'NA'
                            
                            intVariable = intVariable.title() if intVariable != 'NA' and not intVariable.isupper() else intVariable
                            intVarDet = intVarDet.title() if intVarDet != 'NA' else intVarDet
                            if intVariable.lower() in ['pdo', 'amo', 'pdsi']:
                                if intVariable.lower() == 'pdsi':
                                    intVariable = 'Palmer Drought Severity Index'
                                elif intVariable.lower() == 'pdo':
                                    intVariable = 'Pacific Decadal Oscillation'
                                elif intVariable.lower() == 'amo':
                                    intVariable = 'Atlantic Multi-decadal Oscillation'
                                intVariable = intVariable.upper()

                            if intVarDet in interp_det_discard or intVarDet == 'Nan' or intVarDet == 'Na':
                                intVarDet = 'NA'
                            elif intVarDet == 'Surface Temperature' or intVarDet == 'Surface Relative Humidity':
                                intVarDet = 'Surface'
                            else:
                                for name in intVarDet:
                                    if name in interp_ignore_set:
                                        intVarDet = 'NA'
                                        break

                            if 'rank' in inter.keys() :
                                rank = inter['rank']
                            else:
                                rank = inter_len
                
                if infVar != 'NA':
                    df = pd.DataFrame({'coordinates':[geo_coord],'publication':[publication],'filename':[filen], 'archiveType': [archive],'variableType':[vtype], 'rank':[rank],'description':[des],'interpretation/variable':[intVariable],'interpretation/variableDetail':[intVarDet], 'inferredVariable':[infVar], 'inferredVarUnits':[infVarUnits]})
                    inf_table = inf_table.append(df, ignore_index = True) 
    
    table_com=table.explode('proxyObservationType').explode('units').explode('rank').explode('variableType').explode('description').explode('interpretation/variable').explode('interpretation/variableDetail').explode('inferredVariable').explode('inferredVarUnits').reset_index()
    table_com = table_com.drop(columns = ['index'])
    inf_table_com=inf_table.explode('rank').explode('variableType').explode('interpretation/variable').explode('interpretation/variableDetail').explode('inferredVariable').explode('inferredVarUnits').explode('description').reset_index()
    inf_table_com = inf_table_com.drop(columns = ['index'])

    table_com = finalize_data(table_com)
    inf_table_com = finalize_data(inf_table_com)
    
    return table_com, inf_table_com

if __name__ == '__main__':
    initialize_data()
    print(proxyObsTypeutils.proxy_obs_map)
else:
    initialize_data()