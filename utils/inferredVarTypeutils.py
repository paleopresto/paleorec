import re

q_inferred_var = None
inf_named_individuals = None
inf_var_map = {}


def initialize_input_data():
    '''
    q_inferred_var stores the data queried from the linked earth wiki.
    inf_named_individuals stored the data from the linked earth ontology.

    Returns
    -------
    None.

    '''
    
    global q_inferred_var, inf_named_individuals
    
    q_inferred_var = '''
    *Radiocarbon Age
    *D18O
    *Sea Surface Temperature
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
    
    inf_named_individuals = '''Arctic Oscillation (AO, AAO)  Atlantic Multi-decadal Oscillation (AMO)  Bottom water temperature  Carbon dioxide concentration  Carbonate ion concentration  Carbonate saturation  d180  DD  Deuterium excess (excessD)  Free oxygen levels  Julian day  Methane concentration  Moisture content  Nino 1  Nino 1+2  Nino 2  Nino 3  Nino 3.4  Nino 4  Nitrous oxide concentration  North Atlantic Oscillation (NAO)  Ocean mixed layer temperature  Palmer Drought Index  Palmer Drought Severity Index (PDSI)  PH  Precipitation amount  Radiocarbon age  Salinity  Sea surface temperature  Southern Annular Mode (SAM)  Southern oscillation index (SOI)  Surface air temperature  Temperature'''
    inf_named_individuals = inf_named_individuals.split('  ')


def create_inf_var_map():
    '''
    Clean the input data from wiki and the ontology to develop a mapping of given input to cleaned output.

    Returns
    -------
    None.

    '''
    
    global inf_var_map
    
    # MANUAL TASK - ADD THE BASIC STRINGS TO THE MAP.
    
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
    '''
    This method returns the corresponding mapping for the input string or the nearest corresponding value to the values in the cleaned data.
    It uses the inf_var_map created using the ontology and the information from the wiki.
    Given the input string it either finds the mapping in the inf_var_map or
    finds if partial string from the input string is present in inf_var_map, it returns the corresponding result along with the remaining unused string.

    Parameters
    ----------
    vname : string
        Inferred Variable Type being read from the LiPD file currently being processed.

    Returns
    -------
    pred : string
        Result from prediction using the inf_var_map or prediction using part of the input string
    rem : string
        Remaining part of input string if partially used for the prediction else 'NA'

    '''
    pred, rem = 'NA', 'NA'
    if vname in inf_var_map or vname.title() in inf_var_map:
        pred = vname.title()
    elif 'error' in vname.lower() or 'uncertainty' in vname.lower() or 'sampleid' in vname.lower():
        pass
    elif 'nonreliable' in vname.lower():
        pass
    elif 'sampleid' == vname.lower():
        pass
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

if __name__ == '__main__':
    initialize_input_data()
    create_inf_var_map()
else:
    initialize_input_data()
    create_inf_var_map()