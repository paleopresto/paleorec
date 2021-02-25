import os
import pandas as pd
from collections import defaultdict
import re

q_inferred_var = '''Year
*Radiocarbon Age
*D18O
*Sea Surface Temperature
*Age
*Temperature
*Salinity
*Uncertainty temperature
*Temperature1
*Temperature2
*Temperature3
*Uncertainty temperature1
*Thermocline Temperature
*Sedimentation Rate
*Relative Sea Level
*Sea Surface Salinity
*Subsurface Temperature
*Accumulation rate
*Carbonate Ion Concentration
*Mean Accumulation Rate
*Accumulation rate, total organic carbon
*Accumulation rate, calcium carbonate'''
q_inferred_var = q_inferred_var.split('\n*')
q_inferred_var.sort()

inf_named_individuals = '''Age  Arctic Oscillation (AO, AAO)  Atlantic Multi-decadal Oscillation (AMO)  Bottom water temperature  Carbon dioxide concentration  Carbonate ion concentration  Carbonate saturation  d180  DD  Deuterium excess (excessD)  Free oxygen levels  Julian day  Methane concentration  Moisture content  Nino 1  Nino 1+2  Nino 2  Nino 3  Nino 3.4  Nino 4  Nitrous oxide concentration  North Atlantic Oscillation (NAO)  Ocean mixed layer temperature  Palmer Drought Index  Palmer Drought Severity Index (PDSI)  PH  Precipitation amount  Radiocarbon age  Salinity  Sea surface temperature  Southern Annular Mode (SAM)  Southern oscillation index (SOI)  Surface air temperature  Temperature  Year'''
inf_named_individuals = inf_named_individuals.split('  ')

inf_var_map = {}
inf_var_map['Temperature'] = 'Temperature'
inf_var_map['d18O'] = 'D18O'
for inf_var in q_inferred_var:
    m = re.search(r'^\D+', inf_var)
    if m and m.group() in inf_var_map:
        inf_var_map[inf_var] = m.group()
    elif inf_var == 'D18O':
        inf_var_map[inf_var] = 'D18O'
    elif inf_var.title() in inf_named_individuals or inf_var in inf_named_individuals:
        inf_var_map[inf_var] = inf_var.title()
        inf_var_map[inf_var.title()] = inf_var.title()
    else:
        inf_var_map[inf_var] = inf_var
    
for inf_ni in inf_named_individuals:
    if inf_ni not in inf_var_map and inf_ni.title() not in inf_var_map:
        if '(' in inf_ni:
            before = inf_ni[:inf_ni.index('(')].title()
            inf_var_map[inf_ni] = before + inf_ni[inf_ni.index('(') : ]
        else:
            inf_var_map[inf_ni] = inf_ni.title()

def predict_inf_var_type_from_variable_name(vname):
    pred, rem = 'NA', 'NA'
    if vname in inf_var_map or vname.title() in inf_var_map:
        pred = vname.title()
    else:
        for i in range(len(vname)):
            if vname[:i] in inf_var_map:
                pred = vname[:i]
                rem = vname[i:]
                break
            elif vname[i:] in inf_var_map:
                pred = vname[i:]
                rem = vname[:i]
                break
    return pred, rem