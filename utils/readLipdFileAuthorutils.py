# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:46:47 2021

@author: shrav
"""
import lipd
import pandas as pd
import os
import sys
import collections

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

str_count, dict_count = 0,0
inf_var_units_map, ignore_list = None, None
interp_map = {}
interp_det_discard,interp_ignore_set = set(), set()
author_names_map ,reverse_index_author = {}, {}
seed_author_list = {'Y': {'Yair Rosenthal',
  'Yemane Asmerom',
  'Yiming V. Wang',
  'Youbin Sun',
  'Yuan Lin',
  'Yusuke Yokoyama'},
 'D': {'Da Hodgson',
  'David A. Fisher',
  'David Lund',
  'David W. Lea',
  'David M.W. Evans',
  'Delia W. Oppo',
  'Dierk Hebbeln',
  'Dirk Nürnberg',
  'Dorothy K. Pak',
  'Dorthe Dahl-Jensen',
  },
 'R': {'R.G. Fairbanks',
  'Rainer Zahn',
  'Ralf Tiedemann',
  'Ralph R. Schneider',
  'Reinhild Rosenberg',
  'Richey J.N',
  'Riza Yuliratno Setiawan',
  'Rong Xiang',
  'Rob Dunbar'},
 'W': {'W.L. Prell',
  'Wei',
  'Weiguo Liu',
  'Wenyu Shang',
  'William A. Weinlein',
  'Wolfgang Kuhnt',
  'Wu'},
 'S': {'S Lebreiro',
  'Sylke Draschba',
  'Samantha C. Bova',
  'Severinghaus Jeff',
  'Shirley Van Kreveld',
  'Shu Gao',
  'Stefan Löhr',
  'Stefan Schouten',
  'Stefano Bernasconi',
  'Stephan Hetzinger',
  'Stephan Steinke',
  'Stephanie Heilig',
  'Sun -- marine sediment, coral',
  'Syee Weldeab'},
 'J': {'Jaap S. Sinninghe Damsté',
  'Jaime Frigola',
  'Jacek Raddatz',
  'Jaime Nieto-Moreno',
  'James A.P. Bendle',
  'Jan-Rainer Riethdorf',
  'Jana Zech',
  'Jean Lynch-Stieglitz',
  'Jean-Claude Duplessy',
  'Jean-Louis Turon',
  'Jeroen Groeneveld',
  'Jessica E. Tierney',
  'Jian Xu',
  'Jiaqi Liu',
  'Jimin Yu',
  'Jiménez-Espejo Francisco',
  'Joan O. Grimalt',
  'John A. Barron',
  'John Southon',
  'John Tibby',
  'Jonathan Tyler',
  'Josette Duprat',
  'José Abel Flores',
  'Juan Pablo Belmonte',
  'Juan Pablo Bernal',
  'Juillet-Leclerc',
  'Julene P. Marr',
  'Julian P. Sachs',
  'Julie Kalansky',
  'Jun Cheng',
  'Julia Cole',
  'Jung-Hyun Kim',
  'Jérôme Bonnin',
  'Jürgen Pätzold'},
 'C': {'C Kissel',
  'Caitlin Chazen',
  'Camille Levi',
  'Caren T Herbert',
  'Celia Martín-Puertas',
  'Carin Andersson',
  'Carlos Sancho',
  'Carles Pelejero',
  'Carlos Cacho',
  'Caroline Ummenhofer',
  'Celia Corella',
  'Charles D. Keeling',
  'Chris Turney',
  'Christian M. Zdanowicz',
  'Christina L. Belanger',
  'Christophe Kinnard',
  'Christopher D. Charles',
  'Christopher S. Moses',
  'Claire Waelbroeck',
  'Cornelia Glatz',
  'Cyrus Karas'},
 'P': {'P Oliveira',
  'Patricia Jiménez-Amat',
  'Peer Helmke',
  'Penélope González-Sampériz',
  'Peter Demenocal',
  'Peter J. Müller',
  'Philippe Martinez'},
 'T': {'T Rodrigues',
  'Tas Van Ommen',
  'Thomas Damassa',
  'Tatsuhiko Sakamoto',
  'Tebke Böschen',
  'Teresa Vegas-Vilarrúbia',
  'Thibault De Garidel-Thoron',
  'Thomas Blanz',
  'Thomas F. Stocker',
  'Thomas Larsen',
  'Timothy Herbert',
  'Tudhope'},
 'V': {
  'Vasquez-Bedoya',
  'Vanesa Nieto-Moreno',
  'Veronica Willmott',
  'Victor J. Polyak',
  'Vin Morgan'},
 'A': {'Alan Elcheikh',
  'Alexander Matul',
  'Alan Wanamaker',
  'Ana Moreno'
  'Anais Orsi',
  'Andreas Schmittner',
  'Andrea Bagnato',
  'Angel Mojarro',
  'Ann Holbourn',
  'Annette Bolton',
  'Anderson Cooper',
  'Antoni Rosell-Melé',
  'Asami Tetsuo'},
 'B': {'Belen Martrat',
  'Belén Rodríguez-Fonseca',
  'Brad DeLong',
  'Brad Linsley',
  'Blas Valero-Garcés'
  'Boiseau',
  'Brad E. Rosenheim',
  'Braddock K. Linsley',
  'Bruce Cornuelle'},
 'E': {'Elfi Mollier-Vogel',
  'Ellen R.M. Druffel',
  'Elisabeth De Vernal',
  'Elisabeth Isaksson',
  'Eduardo Calvo Buendía',
  'Elsa Cortijo',
  'Enno Schefuß',
  'Euan Smith',
  'Eva Calvo',
  'Eystein Jansen'},
 'L': {'Lars Max',
  'Laura Sbaffi',
  'Laurent Labeyrie',
  'Lester Lembke-Jene',
  'Linda Heusser',
  'Lionel Carter',
  'Lonnie G. Thompson',
  'Lorenzo Vazquez-Selem',
  'Lorraine E. Lisiecki',
  'Lucas J. Lourens',
  'Luejiang Wang'},
 'G': {'Gemma And Canals',
  'Georgina Falster',
  'Geraldine Jacobsen',
  'Gerold Wefer',
  'Gesine Mollenhauer',
  'Guilderson',
  'Guillaume Leduc',
  'Guoqiang Chu'},
 'Q': {'Qianyu Li', 'Qing Sun', 'Quinn'},
 'X': {'Xavier Crosta', 'Xiaohua Wang'},
 'M': {'M Grosjean',
  'M. Schulz ',
  'Mario Morellón'
  'Mahyar Mohtadi',
  'Manman Xie',
  'Mary B. Pfeiffer',
  'Marcus Regenberg',
  'Maria Smirnova',
  'Mark Altabet',
  'Maria Isabel Herrera',
  'Mark Chapman',
  'Markus Kienast',
  'Marta Casado',
  'Martin Ziegler',
  'Maryline Vautravers',
  'Matthew Lacerra',
  'Matthew S. Lachniet',
  'Matthew W. Schmidt',
  'Maureen E. Raymo',
  'Meimei Liu',
  'Michael Sarnthein',
  'Mike A. Hall',
  'Miquel Grimalt',
  'Miquel Canals',
  'Marta Rodrigo-Gámiz',
  'Mitch Lyle',
  'Matthias Kuhnert',
  'Morten Hald',
  'Moustafa -- coral',
  'Mototaka Nakamura',
  'Mustafa O. Moammar',
  'M. Kucera'},
 'K': {'Karl-Heinz Baumann',
  'Katharina Pahnke',
  'Katharine Grant',
  'Katrine Husum',
  'Katsunori Kimoto',
  'Kay-Christian Emeis',
  'Ken´Ichi Ohkushi',
  'Kim M. Cobb',
  'Kirsten Fahl',
  'Km Saunders',
  'Kristin Doering'},
 'F': {'F Abrantes',
  'Thomas Felis -- coral',
  'Fern T. Gibbons',
  'Francisca Martinez-Ruiz',
  'Francisca Martínez-Ruiz',
  'Francisco J. Sierro',
  'Franck Bassinot',
  'Franco Marcantonio',
  'François Guichard'},
 'Z': {'Zhengyu Liu', 'Zhimin Jian', 'Zinke'},
 'H': {'H B Bartels-Jonsdottir',
  'Haase-Schramm',
  'Hai Cheng',
  'Halimeda Kilbourne',
  'Hans Renssen',
  'Hans-Martin Schulz',
  'Harry Elderfield',
  'Heiss -- coral',
  'Helen C. Bostock',
  'Helge Meggers',
  'Helge W. Arz',
  'Helmut Erlenkeuser',
  'Henning Kuhnert',
  'Hideki Ohshima',
  'Hiroshi Kawamura',
  'Hiroyuki Matsuzaki',
  'Hisashi Yamamoto',
  'Hodaka Kawahata',
  'Holger Kuhlmann',
  'Hollander D.J.',
  'Howard J. Spero'},
 'N': {'Nerilie Abram',
  'Nadine Rippert',
  'Nalan Koç',
  'Nathalie F. Goodkin',
  'Nicholas J. Shackleton',
  'Nicolas Caillon',
  'Neil C. Swart',
  'Nils Andersen'},
 'I': {'I M Gil', 'Isabel Cacho', 'Isla S. Castañeda'},
 'U': {'Ulrich Struck','U. Pflaumann',},
 'O': {'Olivier Marchal', 'Osborne'},
 'Á': {'Ánchel Belmonte'}}



def initialize_data():
    
    global inf_var_units_map, ignore_list, interp_map, interp_det_discard,interp_ignore_set
    
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
    ignore_list = ['age', 'depth', 'year', 'Age', 'Depth', 'Year', 'Not Clear', 'No Isotopic Analyes Here']

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
    interp_map['Pdo'] = 'PDO'
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
    interp_map['Amo'] = 'AMO'
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
    interp_map['Pdsi'] = 'PDSI'
    interp_map['T_Air P_Amount Drought Index Spei'] = 'SPEI'
    interp_map['T_Air P_Amount'] = 'Temperature and Precipitation'
    interp_map['Et'] = 'Evapotranspiration'
    interp_map['Relative Humdity'] = 'Relative Humidity'

    interp_det_discard = {'Air', 'Of Precipitation', 'Tropical Or North Pacific Moisture', 'Precipitation', 'Moisture Source', 'Indian Monsoon Strength', 'In The Venezuelan Andes', 'South China Sea', 'In The Southern Tropical Andes','Continental Sweden', 'Strength   Position Of Aleutian Low', 'Rain', 'D18Op', 'Southern Tibet', 'Qaidam Basin', 'Aleutian Low Westerly Storm Trajectories', 'Moisture', 'Enso Pdo', 'Enso  Pacific Ocean Atmosphere Dynamics ', 'E P Lake Water', 'Summer Monsoon', 'Central Asia', 'Central India', 'At  39 Degrees Lat', 'Coastal Lagoon Water', 'Lake  Winds In Eastern Patagonia', 'Relative Amount Of Winter Snowfall', 'Annual Amount Weighted Precipitation Isotopes', 'Evaporative Enrichment Of Leaf Water', '0 58    0 11Ppt Degrees C', 'Surface Waters Of Polar Origin', 'Amount Of Rainfall Change', 'East Asian Monsoon Strength', 'Variations In Winter Temperature In The Alps', 'Sam', 'Regional And Hemispheric Temperature', 'Monsoon Strength', 'Summer Monsoon Rainfall Amount', 'Australian Indonesian Monsoon Rainfall', '20 Mbsl', 'Seasonal  Annual', 'Seasonal', 'Soil Moisture Stress', 'D18O Of The Leaf Water', 'Soil Moisture', 'Precipitation D18O', 'Modelled Precipitation D18O', 'Leaf Water D18O', 'Maximum Air Temperature  Seasonal', 'Maximum Temperature', 'Relative Humidity', 'Atmospheric Surface Temperature', 'Soil Pdsi Conditions', 'Isotopic Composition Of Summer Rainfall', 'Souther Annular Mode  Sam ', 'Net Primary Productivity'}

    interp_ignore_set = {'Seasonal', 'Annual', 'North', 'South', 'East', 'West', 'Northern', 'Southern', 'Eastern', 'Western', 'Tropical','China', 'India', 'Aleutian', 'Asia', 'Alps', 'Summer', 'Winter', 'Polar', 'Monsoon', 'Central'}

def editDistance(str1, str2, m, n):
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]
 
    # Fill d[][] in bottom up manner
    for i in range(m + 1):
        for j in range(n + 1):
 
            if i == 0:
                dp[i][j] = j    # Min. operations = j
 
            elif j == 0:
                dp[i][j] = i    # Min. operations = i
 
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]

            else:
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert
                                   dp[i-1][j],        # Remove
                                   dp[i-1][j-1])    # Replace
 
    return dp[m][n]

def check_fname(auth_list, rn_list):
    if ('.' in auth_list[0] and rn_list[0][0] == auth_list[0][0]):
        return True
    elif rn_list[0] == auth_list[0]:
        return True
    else:
        matching = 0
        auth_fname, rn_fname = auth_list[0], rn_list[0]
        ap, rp = 0, 0
        while ap < len(auth_fname) and rp < len(rn_fname):
            if auth_fname[ap] == rn_fname[rp]:
                matching += 1
                ap += 1
                rp += 1
            else:
                break
        if matching >= 2:
            return True
        return False

def validate_author_name(auth_name):


    global author_names_map

    if auth_name in author_names_map:
        return author_names_map[auth_name]
    
    auth = auth_name.title()
    auth_list = auth.split()
    auth_lastName = auth_list[-1]

    if auth[0].upper in seed_author_list:
        reference_names = seed_author_list[auth[0].upper()]

        for rn in reference_names:

            rn_list = rn.split()
            rn_lastName = rn_list[-1]
            # complete names are same
            if rn_list == auth_list:
                author_names_map[auth_name] = rn
                break
            # lastNames are completely equal
            elif rn_lastName == auth_lastName:
                if check_fname(auth_list, rn_list):
                    author_names_map[auth_name] = rn
                    break
            # refernce lastName is a substring of input lastName
            elif rn_lastName in auth_lastName:
                if check_fname(auth_list, rn_list):
                    author_names_map[auth_name] = rn
                    break
            # check where the characters of the lastName are different
            if len(auth_lastName) >= 3 and len(rn_lastName) >= 3:     
                edit_dist = editDistance(auth_lastName, rn_lastName, len(auth_lastName), len(rn_lastName))
                if edit_dist <= 2:
                    if check_fname(auth_list, rn_list):
                        author_names_map[auth_name] = rn
                        break
    
    if auth_name not in author_names_map:
        author_names_map[auth_name] = auth
    
    if len(auth_list) == 2 and '.' in auth_list[1] and auth_name.isupper():
        author_names_map[auth_name] = " ".join([auth_list[1], auth_list[0].title()])
    
    return author_names_map[auth_name]

def add_to_reversed_index_author_map(names_list, archive):
    global reverse_index_author

    for a_name in names_list.split(','):
        reverse_index_author.setdefault(a_name.strip(),[]).append(archive)

def finalize_data(dataframe):

    dataframe = dataframe.replace('Nan', 'NA', regex=True)
    dataframe = dataframe.replace('g.cm-2.a-1', 'g/cm2a', regex=True)
    dataframe = dataframe.replace('mcm', 'microm', regex=True)
    dataframe = dataframe.replace('NotApplicable', 'NA', regex=True)

    return dataframe

def generate_author_archive_distribution():
    global reverse_index_author

    for key in reverse_index_author.keys():
        reverse_index_author[key] = dict(collections.Counter(reverse_index_author[key]))

    new_author_arch_k = []
    new_author_arch_v = []
    for k,v in reverse_index_author.items():
        l = []
        for ki, vi in v.items():
            l.append("{}({})".format(ki,vi))
        new_author_arch_k.append(k)
        new_author_arch_v.append(",".join(l))
    
    df = pd.DataFrame(list(zip(new_author_arch_k, new_author_arch_v)), columns =['Author', 'Archives'])
    df.to_csv('author_list.csv', sep = ',', encoding = 'utf-8',index = False)


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
    
    table = pd.DataFrame(columns = ['coordinates','publication','filename','authorName','archiveType', 'variableType','description', 'proxyObservationType','units', 'rank', 'interpretation/variable','interpretation/variableDetail', 'inferredVariable', 'inferredVarUnits'])
    inf_table = pd.DataFrame(columns = ['coordinates','publication','filename','authorName','archiveType', 'variableType','description','interpretation/variable','interpretation/variableDetail', 'inferredVariable', 'inferredVarUnits'])
    author_info = set()
    list_issue = {}

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

        a_name_list = []
        archive = d['archiveType']        
        if 'pub' in d:
            path = d['pub'][0]
            if 'author' in path:
                author_list = path['author']
                print(path['author'])
                extract_author_info_from_path(author_list, a_name_list, archive)
                author_info.add(line)

        if line not in author_info and 'investigator' in d:
            path_investigators = d['investigator']
            print(path_investigators)
            extract_author_info_from_path(path_investigators, a_name_list, archive)

        path = d['paleoData']['paleo0']['measurementTable']['paleo0measurement0']['columns']

        if not a_name_list:
            a_name_list.append('NA')

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
                            
                            intVariable = intVariable.title() if intVariable != 'NA' else intVariable
                            intVarDet = intVarDet.title()

                            if intVarDet in interp_det_discard or intVarDet == 'Nan':
                                intVarDet = 'NA'
                            elif intVarDet == 'Surface Temperature' or intVarDet == 'Surface Relative Humidity':
                                intVarDet = 'Surface'
                            else:
                                for name in intVarDet:
                                    if name in interp_ignore_set:
                                        intVarDet = 'NA'
                                        break
                            
                            intVarDet = intVarDet.title()
                            
                            if infVar == 'NA' and intVariable != 'NA' and intVarDet != 'NA':
                                inf_from_interp = (' ').join([intVarDet, intVariable])
                                infVar = inf_from_interp
                                infVarUnits = inf_var_units_map[infVar] if infVar in inf_var_units_map else 'NA'
                                    

                            if unit != 'NotApplicable' or proxyOType != 'NA':
                                for a_name in a_name_list:
                                    df = pd.DataFrame({'coordinates':[geo_coord],'publication':[publication],'filename':[filen],'authorName':[a_name], 'archiveType': [archive],'variableType':[vtype], 'units':[unit],'description':[des],'proxyObservationType':[proxyOType],'rank':[rank],'interpretation/variable':[intVariable],'interpretation/variableDetail':[intVarDet], 'inferredVariable':[infVar], 'inferredVarUnits':[infVarUnits]})
                                    table = table.append(df, ignore_index = True)
                                    inter_set = True
                                
                if not inter_set and (unit != 'NotApplicable' or proxyOType != 'NA'):
                    for a_name in a_name_list:
                        df = pd.DataFrame({'coordinates':[geo_coord],'publication':[publication],'filename':[filen],'authorName':[a_name], 'archiveType': [archive],'variableType':[vtype], 'units':[unit],'description':[des],'proxyObservationType':[proxyOType],'rank':[rank],'interpretation/variable':[intVariable],'interpretation/variableDetail':[intVarDet], 'inferredVariable':[infVar], 'inferredVarUnits':[infVarUnits]})
                        table = table.append(df, ignore_index = True)
                                
            elif vtype == 'inferred':
                if 'inferredVariableType' in path[key].keys() :
                    infVar = path[key]['inferredVariableType']
                    if infVar in ignore_list:
                        continue
                if infVar == 'NA' and 'variableName' in path[key].keys() :
                    vname = path[key]['variableName']
                    infVar, rem = inferredVarTypeutils.predict_inf_var_type_from_variable_name(vname)
                    if infVar == 'NA' and vname not in ignore_list:
                        infVar = vname
                    elif len(infVar) > 45:
                        infVar = 'NA'
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
                                print('Couldn\'t find in map, {}'.format(intVariable))
                                for name in intVariable.split(' '):
                                    if name in interp_ignore_set:
                                        accept = True
                                        intVariable = 'NA'
                                        break
                                if accept:
                                    pass
                                else:
                                    intVariable = 'NA'
                            
                            intVariable = intVariable.title() if intVariable != 'NA' else intVariable
                            intVarDet = intVarDet.title()

                            if intVarDet in interp_det_discard or intVarDet == 'Nan':
                                intVarDet = 'NA'
                            elif intVarDet == 'Surface Temperature' or intVarDet == 'Surface Relative Humidity':
                                intVarDet = 'Surface'
                            else:
                                for name in intVarDet:
                                    if name in interp_ignore_set:
                                        intVarDet = 'NA'
                                        break
                            
                            intVarDet = intVarDet.title()

                            if 'rank' in inter.keys() :
                                rank = inter['rank']
                            else:
                                rank = inter_len
                
                if infVar != 'NA':
                    for a_name in a_name_list:
                        df = pd.DataFrame({'coordinates':[geo_coord],'publication':[publication],'filename':[filen],'authorName':[a_name], 'archiveType': [archive],'variableType':[vtype], 'rank':[rank],'description':[des],'interpretation/variable':[intVariable],'interpretation/variableDetail':[intVarDet], 'inferredVariable':[infVar], 'inferredVarUnits':[infVarUnits]})
                        inf_table = inf_table.append(df, ignore_index = True)
    
    table_com=table.explode('authorName').explode('proxyObservationType').explode('units').explode('rank').explode('variableType').explode('description').explode('interpretation/variable').explode('interpretation/variableDetail').explode('inferredVariable').explode('inferredVarUnits').reset_index()
    table_com = table_com.drop(columns = ['index'])
    inf_table_com=inf_table.explode('authorName').explode('rank').explode('variableType').explode('interpretation/variable').explode('interpretation/variableDetail').explode('inferredVariable').explode('inferredVarUnits').explode('description').reset_index()
    inf_table_com = inf_table_com.drop(columns = ['index'])

    table_com = finalize_data(table_com)
    inf_table_com = finalize_data(inf_table_com)

    generate_author_archive_distribution()
    print('*********************************** LIST ISSUE **************************************')
    print(list_issue)
    
    print('Str type:', str_count)
    print('dict type:', dict_count)

    return table_com, inf_table_com

def extract_author_info_from_path(author_list, a_name_list, archive):
    global str_count, dict_count

    if type(author_list) == str:
        print(type(author_list))
        str_count += 1
        parse_author_string(author_list, a_name_list, archive)
    else:
        dict_count += 1        
        for auth in author_list:
            if 'name' in auth:
                if type(auth['name']) == list:
                    for item in auth['name']:
                        parse_author_string(item, a_name_list, archive)
                else:
                    parse_author_string(auth['name'], a_name_list, archive)

def parse_author_string(author_list, a_name_list, archive):
    author_list = author_list.replace(' abd ', ' AND ')
    author_list = author_list.replace(' ABD ', ' AND ')
    author_list = author_list.replace(' and ', ' AND ')
    semicolon_sep_list = author_list.split(';')
    and_sep_list = []
    for part in semicolon_sep_list:
        and_sep_list.extend(part.strip().split('AND'))

    for part in and_sep_list:
        part = part.strip(' ,')
        p = part.split(',')
        if len(p) == 2:
            new_part = p[1].strip(' ,') + ' ' + p[0].strip(' ,')
        elif len(p) == 4:
            new_part = p[1].strip(' ,') + ' ' + p[0].strip(' ,')
            new_part = validate_author_name(new_part)
            a_name_list.append(new_part)
            add_to_reversed_index_author_map(new_part, archive)

            new_part = p[3].strip(' ,') + ' ' + p[2].strip(' ,')
        elif len(p) > 2:
            f_part = p[1:]
            l_part = p[0]
            f_part.extend(l_part)
            new_part = ' '.join(f_part)
        else:
            if part.endswith('.')  and ' ' in part:
                new_part = ' '.join([part[part.index(' ') + 1 :] ,part[:part.index(' ')]])
            else:
                new_part = part
        new_part = validate_author_name(new_part)
        a_name_list.append(new_part)
        add_to_reversed_index_author_map(new_part, archive)

if __name__ == '__main__':
    initialize_data()
else:
    initialize_data()