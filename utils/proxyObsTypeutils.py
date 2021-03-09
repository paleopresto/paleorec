import os
import pandas as pd
import lipd
import json
from collections import defaultdict
import re

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
q_proxy_obs = q_proxy_obs.split('\n*')

# FOR WINDOWS
periodic_table_path = '..\PeriodicTableJSON.json'
# FOR LINUX
periodic_table_path = '../PeriodicTableJSON.json'

with open(periodic_table_path, 'r', encoding="utf8") as jsonfile:
    element_json = json.load(jsonfile)

periodic_table_elements, periodic_table_name = [], []
for ele in element_json['elements']:
    periodic_table_elements.append(ele['symbol'])
    periodic_table_name.append(ele['name'])

proxy_obs_map = {}
proxy_obs_map['Calcification'] = 'Calcification'
proxy_obs_map['Trsgi'] = 'Trsgi'
proxy_obs_map['C37.concentration'] = '37:2AlkenoneConcentration'
proxy_obs_map['trsgi'] = 'Trsgi'
proxy_obs_map['d-excess'] = 'deuteriumExcess'
proxy_obs_map['deuteriumExcess'] = 'deuteriumExcess'
proxy_obs_map['d2H'] = 'dD'
proxy_obs_map['dD'] = 'dD'
proxy_obs_map['d18o'] = 'd18O'
unknown_proxy = set()

for proxy in q_proxy_obs:
    m = re.search(r'^\D+', proxy)
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
    elif m and m.group() in proxy_obs_map:
            proxy_obs_map[proxy] = m.group()
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


def predict_proxy_obs_type_from_variable_name(vname):
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