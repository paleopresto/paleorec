import json
import re
from sys import platform as _platform
import os

named_individuals, q_proxy_obs = None, None
periodic_table_elements, periodic_table_name = [], []
proxy_obs_map = {}
ignore_set = set()
unknown_proxy = set()

def initialize_input_data():
    '''
    q_proxy_obs stores the data queried from the linked earth wiki.
    named_individuals stores the data from the linked earth ontology.

    Returns
    -------
    None.

    '''

    global named_individuals, q_proxy_obs
    named_individuals = 'Al/Ca  Ar-Ar  B/Ca  Ba/Ca  C  Clay fraction  Color  d13C  d15N  d170  d180  d34S  dD  Density  Diffuse spectral reflectance  Faunal  Fe/Ca  Floral  Grain size  Historic  Layer thickness  Lead Isotope  Li/Ca  Lithics  Luminescence  Magnetic susceptibility  Mg/Ca  Mineral matter  Mn/Ca  Moisture Content  N  Neodymium  Organic matter  P  Permeability  Porosity  Radiocarbon  Resistivity  Sand fraction  Si  Silt fraction  Sr/Ca  TEX86  U-Th  Uk37\'  Uk37  X-Ray diffraction  X-ray fluorescence  Zn/Ca'
    named_individuals = set(named_individuals.split('  '))

    q_proxy_obs = '''DiffuseSpectralReflectance
    *JulianDay
    *Al/Ca
    *B/Ca
    *Ba/Ca
    *Mn/Ca
    *Sr/Ca
    *Zn/Ca
    *Radiocarbon
    *D18O
    *Mg/Ca
    *TEX86
    *TRW
    *Dust
    *Chloride
    *Sulfate
    *Nitrate
    *D13C
    *Depth
    *Age
    *Mg
    *Floral
    *DD
    *C
    *N
    *P
    *Si
    *Uk37
    *Uk37Prime
    *Density
    *GhostMeasured
    *Trsgi
    *Mg Ca
    *SampleCount
    *Segment
    *RingWidth
    *Residual
    *ARS
    *Corrs
    *RBar
    *SD
    *SE
    *EPS
    *Core
    *Uk37prime
    *Upper95
    *Lower95
    *Year old
    *Thickness
    *Na
    *DeltaDensity
    *Reflectance
    *BlueIntensity
    *VarveThickness
    *Reconstructed
    *AgeMin
    *AgeMax
    *SampleID
    *Depth top
    *Depth bottom
    *R650 700
    *R570 630
    *R660 670
    *RABD660 670
    *WaterContent
    *C N
    *BSi
    *MXD
    *EffectiveMoisture
    *Pollen
    *Precipitation
    *Unnamed
    *Sr Ca
    *Calcification1
    *Calcification2
    *Calcification3
    *CalcificationRate
    *Composite
    *Calcification4
    *Notes
    *Notes1
    *Calcification5
    *Calcification
    *Calcification6
    *Calcification7
    *Trsgi1
    *Trsgi2
    *Trsgi3
    *Trsgi4
    *IceAccumulation
    *F
    *Cl
    *Ammonium
    *K
    *Ca
    *Duration
    *Hindex
    *VarveProperty
    *X radiograph dark layer
    *D18O1
    *SedAccumulation
    *Massacum
    *Melt
    *SampleDensity
    *37:2AlkenoneConcentration
    *AlkenoneConcentration
    *AlkenoneAbundance
    *BIT
    *238U
    *Distance
    *232Th
    *230Th/232Th
    *D234U
    *230Th/238U
    *230Th Age uncorrected
    *230Th Age corrected
    *D234U initial
    *TotalOrganicCarbon
    *CDGT
    *C/N
    *CaCO3
    *PollenCount
    *Weight
    *DryBulkDensity
    *37:3AlkenoneConcentration
    *Min sample
    *Max sample
    *Age uncertainty
    *Is date used original model
    *238U content
    *238U uncertainty
    *232Th content
    *232Th uncertainty
    *230Th 232Th ratio
    *230Th 232Th ratio uncertainty
    *230Th 238U activity
    *230Th 238U activity uncertainty
    *Decay constants used
    *Corrected age
    *Corrected age unceratainty
    *Modern reference
    *Al
    *S
    *Ti
    *Mn
    *Fe
    *Rb
    *Sr
    *Zr
    *Ag
    *Sn
    *Te
    *Ba
    *NumberOfObservations
    *Diffuse spectral reflectance
    *Total Organic Carbon
    *BSiO2
    *CalciumCarbonate
    *WetBulkDensity'''
    q_proxy_obs = q_proxy_obs.split('\n    *')

def get_periodic_elements():
    '''
    Get the atomic number and the atomic names for the elements from the periodic table from the file PeriodicTableJSON

    Returns
    -------
    None.

    '''

    global periodic_table_elements, periodic_table_name
    if _platform == "win32":
        print(os.getcwd())
        periodic_table_path = '..\\utils\\PeriodicTableJSON.json'
    else:
        periodic_table_path = '../utils/PeriodicTableJSON.json'

    with open(periodic_table_path, 'r', encoding="utf8") as jsonfile:
        element_json = json.load(jsonfile)


    for ele in element_json['elements']:
        periodic_table_elements.append(ele['symbol'])
        periodic_table_name.append(ele['name'])


def manual_additions_to_map():
    global proxy_obs_map, ignore_set

    # MANUAL ADDITIONS TO THE PROXY OBS MAP
    proxy_obs_map['Calcification'] = 'Calcification'
    proxy_obs_map['Trsgi'] = 'Tree Ring Standardized Growth Index'
    proxy_obs_map['C37.concentration'] = '37:2AlkenoneConcentration'
    proxy_obs_map['trsgi'] = 'Tree Ring Standardized Growth Index'
    proxy_obs_map['d-excess'] = 'D-Excess'
    proxy_obs_map['deuteriumExcess'] = 'D-Excess'
    proxy_obs_map['Deuteriumexcess'] = 'D-Excess'
    proxy_obs_map['d2H'] = 'Dd'
    proxy_obs_map['dD'] = 'Dd'
    proxy_obs_map['d18o'] = 'd18O'
    proxy_obs_map['d18O1'] = 'd18O'
    proxy_obs_map['Mgca'] = 'Mg/Ca'
    proxy_obs_map['Blueintensity'] = 'Blue Intensity'
    proxy_obs_map['MXD'] = 'maximum latewood density'
    proxy_obs_map['TRW'] = 'Tree Ring Width'
    proxy_obs_map['Watercontent'] = 'Water Content'
    proxy_obs_map['Samplecount'] = 'Sample Count'
    proxy_obs_map['Ringwidth'] = 'Tree Ring Width'
    proxy_obs_map['Effectivemoisture'] = 'Effective Moisture'
    proxy_obs_map['EPS'] = 'Expressed Population Signal'
    proxy_obs_map['TOC'] = 'Total Organic Carbon'
    proxy_obs_map['TN'] = 'Total Nitrogen'
    proxy_obs_map['Laminathickness'] = 'Lamina Thickness'
    proxy_obs_map['Foram.Abundance'] = 'Foraminifera Abundance'
    proxy_obs_map['SE'] = 'Standard Error'
    proxy_obs_map['Bsi'] = 'Biogenic Silica'
    proxy_obs_map['Massacum'] = 'Mass Flux'
    proxy_obs_map['R650_700'] = 'Trough area between 650 and 700 nm wavelength'
    proxy_obs_map['R570_630'] = 'Ratio between reflectance at 570 and 630 nm wavelength'
    proxy_obs_map['R660_670'] = 'Ratio between reflectance at 660 and 670 nm wavelength'
    proxy_obs_map['RABD660_670'] = 'relative absorption band depth from 660 to 670 nm'
    proxy_obs_map['ARS'] = 'ARSTAN chronology'
    proxy_obs_map['Rbar'] = 'Rbar (mean pair correlation)'
    proxy_obs_map['D50'] = 'Median, grain size (D50)'
    proxy_obs_map['Grainsizemode'] = 'Mode, grain size'
    proxy_obs_map['DBD'] = 'Dry Bulk Density'
    proxy_obs_map['Dry Bulk Density'] = 'Dry Bulk Density'
    proxy_obs_map['Brgdgt'] = 'brGDGT'
    proxy_obs_map['Brgdgtiiia'] = 'brGDGT'
    proxy_obs_map['Brgdgtiiib'] = 'brGDGT'
    proxy_obs_map['Brgdgtiia'] = 'brGDGT'
    proxy_obs_map['Brgdgtiib'] = 'brGDGT'
    proxy_obs_map['Brgdgtia'] = 'brGDGT'
    proxy_obs_map['Brgdgtib'] = 'brGDGT'
    proxy_obs_map['N C24'] = 'n-alkane 24 carbon chain'
    proxy_obs_map['N C26'] = 'n-alkane 26 carbon chain'
    proxy_obs_map['N C28'] = 'n-alkane 28 carbon chain'
    proxy_obs_map['IRD'] = 'Ice-rafted debris'


    ignore_set = {'Upper95', 'Lower95', 'Sampleid', 'Julianday', 'SD', 'Elevation Sample', 'Repeats', 'Age', 'age', 'Year', 'year', 'Depth', 'depth', 'Hindex', 'Stdev C24', 'Stdev C26', 'Stdev C28', 'Surface.Temp'}


def create_proxy_obs_map():
    '''
    Clean the input data from wiki and the ontology to develop a mapping of given input to cleaned output.
    Several checks are involved. If the input has length <= 2 then it is checked if it is an element in the periodic table
    Other reductions are made to create a particular way of writing data example d13C = D13C, Mg_Ca = Mg/Ca

    Returns
    -------
    None.

    '''
    global proxy_obs_map, unknown_proxy

    for proxy in q_proxy_obs:
        if proxy in proxy_obs_map or proxy.title() in proxy_obs_map or proxy.lower() in proxy_obs_map or proxy.upper() in proxy_obs_map:
            if proxy.title() in proxy_obs_map:
                proxy_obs_map[proxy] = proxy_obs_map[proxy.title()]
            elif proxy.lower() in proxy_obs_map:
                proxy_obs_map[proxy] = proxy_obs_map[proxy.lower()]
            elif proxy.upper() in proxy_obs_map:
                proxy_obs_map[proxy] = proxy_obs_map[proxy.upper()]
            continue
        elif 'Uk37' in proxy:
            pass
        elif proxy[-1].isdigit() and proxy[:-1] in proxy_obs_map:
            proxy_obs_map[proxy] = proxy_obs_map[proxy[:-1]]
            continue
        elif (not proxy.islower() and not proxy.isupper()) and '/' not in proxy and ':' not in proxy:
            new_proxy = []
            for c in proxy:
                if c.isupper():
                    new_proxy.append(' ')
                new_proxy.append(c)

            new_proxy = ''.join(new_proxy).strip()
            in_periodic = False
            for n in new_proxy.split(' '):
                if n in periodic_table_elements:
                    in_periodic = True
                    break
            if not in_periodic:
                if new_proxy in proxy_obs_map:
                    pass
                else:
                    proxy_obs_map[proxy] = new_proxy
                    proxy_obs_map[new_proxy] = new_proxy
                continue

        if len(proxy) <= 2:
            if proxy in periodic_table_elements:
                proxy_obs_map[proxy] = proxy
            else:
                unknown_proxy.add(proxy)
        elif '/' in proxy:
            s = proxy.split('/')
            if (s[1] == 'Ca' or s[1] == 'N') and s[0] in periodic_table_elements:
                proxy_obs_map[proxy] = proxy
            elif 'U' in s[0] and 'U' in s[1] or 'Th' in s[0] and 'Th' in s[1]:
                proxy_obs_map[proxy] = proxy
            else:
                unknown_proxy.add(proxy)
        elif proxy.startswith('D') or proxy.startswith('d'):
            if proxy.isalpha():
                proxy_obs_map[proxy] = proxy
            else:
                proxyl = str('d' + proxy[1:])
                if proxyl in named_individuals:
                    proxy_obs_map[proxy] = proxyl
                    proxy_obs_map[proxyl] = proxyl
                else:
                    proxy_obs_map[proxy] = proxyl
                    proxy_obs_map[proxyl] = proxyl
                    unknown_proxy.add(proxy)
        elif proxy.startswith('R') and proxy.replace(' ', '_') in proxy_obs_map:
            proxy_obs_map[proxy] = proxy_obs_map[proxy.replace(' ', '_')]
        elif ' ' in proxy:
            s = proxy.split(' ')
            if len(s) > 2:
                proxy_obs_map[proxy] = proxy
            elif len(s[1]) > 2:
                proxy_obs_map[proxy] = proxy
            else:
                num = s[0].title()
                den = s[1].title()
                if den in periodic_table_elements and str(num + '/' + den) in q_proxy_obs:
                    proxy_obs_map[proxy] = str(num + '/' + den)
        elif 'prime' in proxy or 'Prime' in proxy:
            if 'prime' in proxy:
                s = proxy.split('prime')
            else:
                s = proxy.split('Prime')
            if s[0] in named_individuals:
                proxy_obs_map[proxy] = str(s[0] + "\'")
        elif proxy.lower() in named_individuals or proxy in named_individuals:
            proxy_obs_map[proxy] = proxy if proxy in named_individuals else proxy.lower()
        else:
            proxy_obs_map[proxy] = proxy
            unknown_proxy.add(proxy)

    # print(proxy_obs_map)



def predict_proxy_obs_type_from_variable_name(vname):
    '''
    This method returns the corresponding mapping for the input string or the nearest corresponding value to the values in the cleaned data.
    It uses the proxy_obs_map created using the ontology and the information from the wiki.
    Given the input string it either finds the mapping in the proxy_obs_map or
    finds if partial string from the input string is present in proxy_obs_map, it returns the corresponding result along with the remaining unused string.

    Parameters
    ----------
    vname : string
        Proxy Observation Type being read from the LiPD file currently being processed.

    Returns
    -------
    pred : string
        Result from prediction using the proxy_obs_type or prediction using part of the input string
    rem : string
        Remaining part of input string if partially used for the prediction else 'NA'

    '''
    pred, rem = 'NA', 'NA'
    if vname in proxy_obs_map:
        pred = proxy_obs_map[vname]
    elif vname.title() in proxy_obs_map:
        pred = proxy_obs_map[vname.title()]
    elif vname.isupper():
        pass
    elif '\'' in vname:
        pred = vname
    elif 'bubbleNumberDensity' in vname:
        proxy_obs_map[vname] = vname
        pred = vname
    elif vname in periodic_table_name:
        pred = vname
    elif '_' in vname:
        if 'Ca' in vname or 'N' in vname:
            vname = vname.replace('_', '/')
            pred = proxy_obs_map.get(vname, 'NA')
        else:
            vname = vname.replace('_', ' ')
            pred = proxy_obs_map.get(vname, 'NA')
    elif 'Planktonic.' in vname or 'Benthic.' in vname or 'planktonic.' in vname or 'benthic.' in vname or 'planktic' in vname or 'Planktic' in vname:
            ind = vname.index('.')
            pred = vname[ind+1:]
            rem = vname[:ind]
    else:
        for i in range(len(vname)):
            if vname[:i] in proxy_obs_map:
                pred = vname[:i]
                rem = vname[i:]
                break
            elif vname[i:] in proxy_obs_map:
                pred = vname[i:]
                rem = vname[:i]
                break
        if pred == 'Ca' and len(rem) > 2:
            possible_proxy = rem[-2:] + '/' + pred
            if possible_proxy in proxy_obs_map:
                pred = possible_proxy
                rem = rem[:-2]
    return pred, rem

def validate_proxyObsType(proxyObsType):

    if type(proxyObsType) != list:
        
        if 'error' in proxyObsType.lower():
            return 'NA'


        if 'Depth' in proxyObsType.title() or 'Latitude' in proxyObsType.title() or 'Longitude' in proxyObsType.title():
            return 'NA'
        elif proxyObsType in ignore_set or proxyObsType.title() in ignore_set:
            return 'NA'
        elif proxyObsType in proxy_obs_map:
            return proxy_obs_map[proxyObsType]
        elif proxyObsType.title() in proxy_obs_map:
            return proxy_obs_map[proxyObsType.title()]

    return proxyObsType

if __name__ == '__main__':
    initialize_input_data()
    get_periodic_elements()
    manual_additions_to_map()
    create_proxy_obs_map()
else:
    initialize_input_data()
    get_periodic_elements()
    manual_additions_to_map()
    create_proxy_obs_map()
