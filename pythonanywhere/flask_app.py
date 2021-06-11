# A very simple Flask Hello World app for you to get started with...

from flask import Flask, request
from flask import make_response, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import re
# from misc import get_dsn
import json
# from lpd_noaa import LPD_NOAA
# from jsons import idx_num_to_name
# from csvs import merge_csv_metadata
# from loggers import create_logger
# from linkedearth import wiki_query

from MCpredict import MCpredict
from LSTMpredict import LSTMpredict
import os
import glob
from pydash import py_


# logger_flask = create_logger("flask")

app = Flask(__name__)
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["5000 per day", "500 per hour"]
)
# flask_dir = "/home/cheiser/mysite/"
flask_dir = ''
model_mc_file_path=''
pred3MC = MCpredict(3, 5, model_file_path=flask_dir, ground_truth_path=flask_dir)
pred4MC = MCpredict(4, 5, model_file_path=flask_dir, ground_truth_path=flask_dir)

predLSTM = LSTMpredict(model_file_path=flask_dir, ground_truth_file_path=flask_dir, topk=5)
archives_for_MC = {}
autocomplete_file_path = None

time_map = {'age' : ['year BP', 'cal year BP', 'ky BP', 'my BP'], 
    'year' : ['year CE','year AD']}

# other units for 
# 'age' : ['year B.P.','yr B.P.','yr BP','BP','yrs BP','years B.P.', 'yr. BP','yr. B.P.', 'cal. BP', 'cal B.P.', 'year BP','years BP']
# 'kage' : ['kyr BP','kaBP','ky','kyr','kyr B.P.', 'ka B.P.', 'ky BP', 'kyrs BP','ky B.P.', 'kyrs B.P.', 'kyBP', 'kyrBP']
# 'year' : ['AD','CE','year C.E.','year A.D.','years C.E.','years A.D.','yr CE','yr AD','yr C.E.','yr A.D.', 'yrs C.E.', 'yrs A.D.', 'yrs CE', 'yrs AD']
# 'mage' : ['myr BP', 'myrs BP', 'ma BP', 'ma','my B.P.', 'myr B.P.', 'myrs B.P.', 'ma B.P.']

def get_average_half_len_for_autocomplete():
    """
    names_set_dict from predict_object in LSTM contains the list of all the possible values for each fieldType
    This method calculates half of the average length of value for each fieldType.
    
    Possible use case during autocomplete search, when a user enters half the characters for a fieldType, 
    we can use edit distance to get the most similar words to the provided word.

    Returns
    -------
    avg_half_len_map : dict
        Stores the average half length for each fieldType.

    """
    avg_half_len_map = {}
    for key, in_set in predLSTM.names_set.items():
        sum_len_set = sum([len(word) for word in in_set])
        avg_half_len_map[key] = sum_len_set//(len(in_set) * 2)
    return avg_half_len_map

avg_half_len_map = get_average_half_len_for_autocomplete()

def get_latest_file_with_path(path, *paths):
    '''
    Method to get the full path name for the latest file for the input parameter in paths.
    This method uses the os.path.getctime function to get the most recently created file that matches the filename pattern in the provided path. 

    Parameters
    ----------
    path : string
        Root pathname for the files.
    *paths : string list
        These are the var args field, the optional set of strings to denote the full path to the file names.

    Returns
    -------
    latest_file : string
        Full path name for the latest file provided in the paths parameter.

    '''
    
    fullpath = os.path.join(path, *paths)
    list_of_files = glob.iglob(fullpath)  
    if not list_of_files:                
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

ground_truth_file = get_latest_file_with_path('', 'ground_truth_label_*.json')
with open(ground_truth_file, 'r') as json_file:
    ground_truth_dict = json.load(json_file)

archives_map = ground_truth_dict['archives_map']
names_set_ind_map = {'proxyObservationType' : 1, 'proxyObservationTypeUnits' : 2, 'interpretation/variable' : 3, 'interpretation/variableDetail' : 4, 'inferredVariable' : 5, 'inferredVariableUnits' : 6}
names_set = {}

def load_names_set_from_file(file_name):
    '''
    Method to load the dict containing the list of all possible values for each fieldType, used for autocomplete suggestions.

    Parameters
    ----------
    file_name : string
        File containing the data for autocomplete suggestions.

    Returns
    -------
    None.

    '''
    global names_set
    with open(file_name, 'r', encoding='utf-8') as autocomplete_file_:
        names_set = json.load(autocomplete_file_)

@app.route('/test', methods=["GET"])
@limiter.exempt
def _test():
    #logger_flask.info("Flask API Test: Success")
    return "Flask API Test response: Success"

@app.route("/api/wikiquery", methods=["POST"])
@limiter.exempt
def _wiki_query():
    _results = []
    logger_flask.info("Flask: Entering _wiki_query")
    try:
        logger_flask.info("Flask: Creating query string")
        # real testing
        _results = wiki_query(request.json)
        # hard coded data testing
        # _results = wiki_query(opts)

    except Exception as e:
        logger_flask.error("Flask: wiki_query: {}".format(e))
    logger_flask.info("Finished. Sending back formatted query")
    # logger_flask.info(_results)
    return _results;

@app.route('/api/noaa', methods=["POST"])
@limiter.exempt
def _noaa_start():
    out = []
    logger_flask.info("Flask: LiPD Data received...")
    # logger_flask.info(request)
    # TESTER = {'@context': 'context.jsonld', 'archiveType': 'lake sediment', 'dataSetName': 'Castilla.Lane.2009', 'funding': [{'agency': ['National Geographic Society', 'NSF'], 'grant': ['BCS-0550382']}], 'geo': {'geometry': {'coordinates': [-70.88, 18.8, 1005], 'type': 'Point'}, 'properties': {'geometry_type': 'Point', 'politicalUnit': 'Dominican Republic', 'properties_elevation_unit': 'm', 'siteName': 'Laguna de Felipe'}, 'type': 'Feature'}, 'googleDataURL': 'https://docs.google.com/spreadsheets/d/1ZtQEMAdr12saY7Xe-ABc_LE4aeU_zHz3tWZUSSaePq4', 'googleMetadataWorksheet': 'o2s0ffa', 'googleSpreadSheetKey': '1ZtQEMAdr12saY7Xe-ABc_LE4aeU_zHz3tWZUSSaePq4', 'investigators': 'Lane, C.S.; Horn, S.P.; Mora, C.I.; Orvis, K.H.', 'maxYear': 1990.228261, 'metadataMD5': 'e664497687a6446c75319d39eb1a53ad', 'minYear': 596.7162845, 'originalDataURL': 'this study', 'paleoData': OrderedDict([('paleo0', OrderedDict([('measurementTable', OrderedDict([('paleo0measurement0', {'WDSPaleoUrl': 'https://www1.ncdc.noaa.gov/pub/data/paleo/pages2k/nam2k-hydro-v1-1.0.0/data-version-2017/Castilla.Lane.2009.txt', 'columns': OrderedDict([('d18O_calcite', {'QCnotes': 'Biogenic carbonates only present in 2 intervals of core - hence discontinuous record', 'TSid': 'NAm2kHydro060', 'dataType': 'float', 'description': 'oxygen isotopes from biogenic carbonates', 'hasMaxValue': 4.2999999999999998, 'hasMeanValue': 1.7972631578947371, 'hasMedianValue': 1.6795, 'hasMinValue': 0.0030000000000000001, 'hasResolution': {'hasMinValue': 2.1749999999999545, 'hasMaxValue': 311.58199999999999, 'hasMeanValue': 36.888405405405408, 'hasMedianValue': 22.924999999999955}, 'interpretation': [{'basis': 'closed basin lake water should reflect E/P ratio; no modern calibration', 'interpDirection': 'negative', 'local': 'TRUE', 'variable': 'M', 'variableDetail': 'effective'}], 'measurementMaterial': 'Adult monospecific ostracod valves and calcified charophyte oospores', 'number': 1, 'proxy': 'd18O', 'tableMD5': 'ded0c7396301a6f9d57015490e0e3ecd', 'units': 'permil (VPBD)', 'useInNAm2kHydro': 'FALSE', 'variableName': 'd18O_calcite', 'values': [4.3, 2.33, 1.595, 2.67, 2.024, 1.003, 2.292, 2.555, 2.297, 1.34, 1.526, 1.929, 2.539, 2.424, 3.195, 2.803, 2.134, 1.574, 1.84, 0.454, 1.739, 1.584, 0.969, 1.58, 1.899, 1.62, 1.43, 1.883, 1.077, 1.236, 1.423, 2.178, 1.927, 1.61, 1.107, 1.531, 0.676, 0.003]}), ('year', {'TSid': 'PYTA8BLWQIF', 'dataType': 'float', 'description': 'Year AD', 'hasMaxValue': 1825.6579999999999, 'hasMeanValue': 1078.7001842105262, 'hasMedianValue': 1046.4565, 'hasMinValue': 460.78699999999998, 'hasResolution': {'hasMinValue': 'nan', 'hasMaxValue': 'nan', 'hasMeanValue': 'nan', 'hasMedianValue': 'nan'}, 'inferredVariableType': 'year', 'number': 2, 'units': 'AD', 'variableName': 'year', 'variableType': 'inferred', 'values': [1825.658, 1817.792, 1778.341, 1756.697, 1743.71, 1713.409, 1700.422, 1683.107, 1659.299, 1555.408, 1518.613, 1382.256, 1070.674, 1066.868, 1064.149, 1061.974, 1057.625, 1055.45, 1053.275, 1039.638, 993.788, 959.401, 879.164, 856.24, 816.121, 787.465, 764.54, 724.422, 712.96, 672.841, 598.336, 575.411, 558.218, 535.293, 512.368, 495.175, 483.712, 460.787]}), ('age', {'TSid': 'PYT0C4SVLX5', 'dataType': 'float', 'description': 'Years before present (1950) BP', 'hasMaxValue': 1489.213, 'hasMeanValue': 871.29981578947377, 'hasMedianValue': 903.54349999999999, 'hasMinValue': 124.342, 'hasResolution': {'hasMinValue': 'nan', 'hasMaxValue': 'nan', 'hasMeanValue': 'nan', 'hasMedianValue': 'nan'}, 'inferredVariableType': 'age', 'number': 3, 'units': 'BP', 'variableName': 'age', 'variableType': 'inferred', 'values': [124.342, 132.208, 171.659, 193.303, 206.29, 236.591, 249.578, 266.893, 290.701, 394.592, 431.387, 567.744, 879.326, 883.132, 885.851, 888.026, 892.375, 894.55, 896.725, 910.362, 956.212, 990.599, 1070.836, 1093.76, 1133.879, 1162.535, 1185.46, 1225.578, 1237.04, 1277.159, 1351.664, 1374.589, 1391.782, 1414.707, 1437.632, 1454.825, 1466.288, 1489.213]}), ('d13C_calcite', {'TSid': 'NAm2kHydro061', 'dataType': 'float', 'description': 'carbon isotopes from biogenic carbonates', 'hasMaxValue': 4.2279999999999998, 'hasMeanValue': 2.3516944444444445, 'hasMedianValue': 2.577, 'hasMinValue': -0.187, 'hasResolution': {'hasMinValue': 2.1749999999999545, 'hasMaxValue': 447.93900000000002, 'hasMeanValue': 38.996314285714284, 'hasMedianValue': 22.924999999999955}, 'interpretation': [{'variable': 'complicated', 'variableDetail': 'Higher values could be warm and dry, decreased partial pressures of atmospheric CO2, or a shift towards more warm season precip'}], 'measurementMaterial': 'Adult monospecific ostracod valves and calcified charophyte oospores', 'number': 4, 'proxy': 'd13C', 'tableMD5': 'ded0c7396301a6f9d57015490e0e3ecd', 'units': 'permil (VPBD)', 'useInNAm2kHydro': 'FALSE', 'variableName': 'd13C_calcite', 'values': [4.228, 2.351, 2.577, 3.109, 2.742, 2.146, -0.074, 1.095, 1.579, 'nan', -0.187, 'nan', 0.798, 2.43, 3.169, 3.049, 3.27, 2.344, 2.553, 2.909, 3.871, 2.715, 2.715, 2.546, 2.579, 2.577, 2.16, 3.092, 2.6, 1.825, 3.709, 2.798, 3.054, 2.659, 1.708, 2.025, 1.681, 0.259]})]), 'dataMD5': 'ded0c7396301a6f9d57015490e0e3ecd', 'filename': 'Castilla.Lane.2009.paleo0measurement0.csv', 'googleWorkSheetKey': 'opyqd7g', 'missingValue': 'nan', 'number': 4, 'paleoMeasurementTableNumber': 1, 'paleoNumber': 1, 'tableName': 'paleo0measurement0'})]))])), ('paleo1', OrderedDict([('measurementTable', OrderedDict([('paleo1measurement0', {'WDSPaleoUrl': 'https://www1.ncdc.noaa.gov/pub/data/paleo/pages2k/nam2k-hydro-v1-1.0.0/data-version-2017/Castilla.Lane.2009.txt', 'columns': OrderedDict([('d18O_ostracodes', {'TSid': 'NAm2kHydro062', 'dataType': 'float', 'description': 'oxygen isotopes from ostracod valves', 'hasMaxValue': 3.6379999999999999, 'hasMeanValue': -0.16929661016949152, 'hasMedianValue': 0.048000000000000001, 'hasMinValue': -4.5339999999999998, 'hasResolution': {'hasMinValue': 0.66599999999999682, 'hasMaxValue': 332.47500000000002, 'hasMeanValue': 11.910358974358976, 'hasMedianValue': 5.9020000000000001}, 'interpretation': [{'basis': 'closed basin lake water should reflect E/P ratio', 'interpDirection': 'negative', 'local': 'TRUE', 'variable': 'M', 'variableDetail': 'effective'}], 'measurementMaterial': 'Adult Cythridella boldii ostracod valves', 'number': 1, 'proxy': 'd18O', 'tableMD5': '1809df60cd8a02751c440d87638adbe6', 'units': 'permil (VPBD)', 'useInNAm2kHydro': 'TRUE', 'variableName': 'd18O_ostracodes', 'values': [0.079, -0.452, 1.189, 0.462, 2.232, -0.783, 0.332, -0.995, -1.565, -2.307, 0.631, 0.937, 1.154, -0.67, 0.102, -1.23, -2.287, -0.024, 1.196, 0.037, 0.856, 0.443, 0.691, 2.02, -0.626, 3.638, 0.694, 0.289, 1.255, -1.76, 0.456, -0.137, 0.926, 1.281, 0.417, -1.353, 0.098, 0.889, -0.115, 1.34, -1.341, 1.229, 0.538, 0.467, -3.659, -2.064, -0.34, -1.432, -1.789, -1.274, -1.715, -3.708, -2.809, -3.835, -4.534, -0.99, -2.162, -2.295, -1.827, -2.17, -2.521, -3.073, -3.439, -1.197, -0.833, -1.057, 0.487, -0.778, -1.564, -1.118, -0.251, 0.825, 0.002, 0.278, -0.203, 3.025, 1.589, 1.38, 0.629, 0.034, 0.443, 0.059, 2.391, 2.797, 2.738, 0.663, 1.75, 1.888, -0.581, 0.174, 0.903, -2.07, 0.441, 0.973, -0.549, -0.542, 0.629, -1.473, -2.258, 1.291, 1.269, -1.217, -0.369, 0.507, -0.49, -0.652, 0.771, -0.279, 0.777, 2.823, 1.328, 0.304, 0.994, -0.906, 1.544, 0.418, -0.877, -1.434]}), ('year', {'TSid': 'PYT8YL4IS4C', 'dataType': 'float', 'description': 'Year AD', 'hasMaxValue': 1990.2280000000001, 'hasMeanValue': 1597.3161186440677, 'hasMedianValue': 1796.0, 'hasMinValue': 596.71600000000001, 'hasResolution': {'hasMinValue': 'nan', 'hasMaxValue': 'nan', 'hasMeanValue': 'nan', 'hasMedianValue': 'nan'}, 'inferredVariableType': 'year', 'number': 2, 'units': 'AD', 'variableName': 'year', 'variableType': 'inferred', 'values': [1990.228, 1968.587, 1964.652, 1960.717, 1954.815, 1948.913, 1944.978, 1939.076, 1933.174, 1925.304, 1917.435, 1913.5, 1909.565, 1905.63, 1901.696, 1897.761, 1891.859, 1878.087, 1866.283, 1852.511, 1846.609, 1842.674, 1838.739, 1834.804, 1828.902, 1823.0, 1822.0, 1821.0, 1820.333, 1819.667, 1819.0, 1818.333, 1817.667, 1817.0, 1815.667, 1815.0, 1814.333, 1813.667, 1813.0, 1811.333, 1810.333, 1809.667, 1809.0, 1807.667, 1807.0, 1806.333, 1805.333, 1804.333, 1803.667, 1802.333, 1801.667, 1801.0, 1800.333, 1799.667, 1799.0, 1798.333, 1797.667, 1797.0, 1796.333, 1795.667, 1795.0, 1789.72, 1784.44, 1779.16, 1773.88, 1768.6, 1763.32, 1755.4, 1750.12, 1721.08, 1710.52, 1699.96, 1692.04, 1678.84, 1668.28, 1655.08, 1641.88, 1633.96, 1626.04, 1620.76, 1615.48, 1602.28, 1591.72, 1578.52, 1573.24, 1562.68, 1557.4, 1549.48, 1519.319, 1503.745, 1464.809, 1453.128, 1445.34, 1437.553, 1418.085, 1359.681, 1351.894, 1342.489, 1325.957, 993.482, 964.092, 934.702, 920.007, 905.312, 875.922, 817.142, 802.447, 787.752, 773.057, 758.362, 743.667, 728.972, 714.277, 699.582, 684.887, 640.801, 611.411, 596.716]}), ('age', {'TSid': 'PYTHGAME4YN', 'dataType': 'float', 'description': 'Years before present (1950) BP', 'hasMaxValue': 1353.2840000000001, 'hasMeanValue': 352.68388135593216, 'hasMedianValue': 154.0, 'hasMinValue': -40.228000000000002, 'hasResolution': {'hasMinValue': 'nan', 'hasMaxValue': 'nan', 'hasMeanValue': 'nan', 'hasMedianValue': 'nan'}, 'inferredVariableType': 'age', 'number': 3, 'units': 'BP', 'variableName': 'age', 'variableType': 'inferred', 'values': [-40.228, -18.587, -14.652, -10.717, -4.815, 1.087, 5.022, 10.924, 16.826, 24.696, 32.565, 36.5, 40.435, 44.37, 48.304, 52.239, 58.141, 71.913, 83.717, 97.489, 103.391, 107.326, 111.261, 115.196, 121.098, 127.0, 128.0, 129.0, 129.667, 130.333, 131.0, 131.667, 132.333, 133.0, 134.333, 135.0, 135.667, 136.333, 137.0, 138.667, 139.667, 140.333, 141.0, 142.333, 143.0, 143.667, 144.667, 145.667, 146.333, 147.667, 148.333, 149.0, 149.667, 150.333, 151.0, 151.667, 152.333, 153.0, 153.667, 154.333, 155.0, 160.28, 165.56, 170.84, 176.12, 181.4, 186.68, 194.6, 199.88, 228.92, 239.48, 250.04, 257.96, 271.16, 281.72, 294.92, 308.12, 316.04, 323.96, 329.24, 334.52, 347.72, 358.28, 371.48, 376.76, 387.32, 392.6, 400.52, 430.681, 446.255, 485.191, 496.872, 504.66, 512.447, 531.915, 590.319, 598.106, 607.511, 624.043, 956.518, 985.908, 1015.298, 1029.993, 1044.688, 1074.078, 1132.858, 1147.553, 1162.248, 1176.943, 1191.638, 1206.333, 1221.028, 1235.723, 1250.418, 1265.113, 1309.199, 1338.589, 1353.284]}), ('d13C_ostracodes', {'TSid': 'NAm2kHydro063', 'dataType': 'float', 'description': 'carbon isotopes from ostracod valves', 'hasMaxValue': -0.44, 'hasMeanValue': -4.7993389830508484, 'hasMedianValue': -5.1649999999999991, 'hasMinValue': -8.9700000000000006, 'hasResolution': {'hasMinValue': 0.66599999999999682, 'hasMaxValue': 332.47500000000002, 'hasMeanValue': 11.910358974358976, 'hasMedianValue': 5.9020000000000001}, 'measurementMaterial': 'Adult Cythridella boldii ostracod valves', 'number': 4, 'proxy': 'd13C', 'tableMD5': '1809df60cd8a02751c440d87638adbe6', 'units': 'permil (VPBD)', 'useInNAm2kHydro': 'FALSE', 'variableName': 'd13C_ostracodes', 'values': [-5.575, -5.265, -4.574, -3.519, -3.395, -4.5, -4.302, -5.282, -6.837, -6.779, -3.748, -1.906, -1.84, -2.576, -0.932, -1.512, -2.9, -1.826, -5.128, -6.413, -5.612, -5.835, -5.716, -5.169, -7.808, -5.508, -4.234, -5.758, -5.331, -6.005, -5.908, -4.616, -5.071, -5.44, -6.107, -5.612, -5.161, -5.774, -6.567, -6.42, -7.37, -5.537, -5.388, -7.269, -6.876, -6.741, -5.527, -5.083, -6.367, -6.473, -6.006, -6.6, -5.492, -6.192, -6.371, -4.97, -5.106, -5.658, -5.372, -4.956, -5.899, -5.887, -5.686, -7.28, -5.824, -7.019, -6.212, -6.359, -6.171, -3.201, -6.168, -5.056, -5.316, -5.361, -5.66, -8.341, -8.97, -8.744, -6.875, -6.009, -6.133, -5.016, -4.295, -3.221, -2.083, -4.149, -4.071, -4.894, -5.576, -5.943, -4.477, -3.825, -3.659, -3.696, -4.333, -3.721, -2.848, -4.376, -4.453, -2.36, -0.44, -1.96, -2.43, -2.66, -0.77, -2.01, -3.23, -3.58, -3.23, -3.51, -3.75, -3.1, -2.1, -3.23, -1.69, -2.43, -2.82, -2.4]})]), 'dataMD5': '1809df60cd8a02751c440d87638adbe6', 'filename': 'Castilla.Lane.2009.paleo1measurement0.csv', 'googleWorkSheetKey': 'oc7mjtf', 'missingValue': 'nan', 'number': 4, 'paleoMeasurementTableNumber': 1, 'paleoNumber': 2, 'tableName': 'paleo1measurement0'})]))]))]), 'pub': [{'author': 'Lane, Chad S.;  Horn, Sally P.;  Mora, Claudia I.;  Orvis, Kenneth H.', 'dataUrl': 'doi.org', 'identifier': [{'id': '10.1016/j.quascirev.2009.04.013', 'type': 'doi', 'url': 'http://dx.doi.org/10.1016/j.quascirev.2009.04.013'}], 'issue': '23-24', 'journal': 'Quaternary Science Reviews', 'page': '2239-2260', 'pubDataUrl': 'doi.org', 'pubYear': '2009', 'publisher': 'Elsevier BV', 'title': 'Late-Holocene paleoenvironmental change at mid-elevation on the Caribbean slope of the Cordillera Central, Dominican Republic: a multi-site, multi-proxy analysis', 'type': 'journal-article', 'volume': '28'}], 'studyName': 'Isotope data from Laguna de Felipe recording Little Ice Age Aridity in the Caribbean', 'tagMD5': '285bfb25d48ae8f66366eebf8831cc0a', 'lipdVersion': 1.3}
    # logger_flask.info(request.json)
    logger_flask.info("Flask: Processing: {}".format(request.json["metadata"]["dataSetName"]))
    try:
        logger_flask.info("Flask: Start processing to NOAA...")
        if "csvs" in request.json:
            logger_flask.info("Flask: Csv data exists")
            _csvs = request.json["csvs"]
            logger_flask.info("Flask: Start idx_num_to_name")
            _json = idx_num_to_name(request.json["metadata"])
            logger_flask.info("Flask: Start merge_csv_metadata")
            _json = merge_csv_metadata(_json, _csvs)
        else:
            logger_flask.info("Flask: No CSV data provided : Quitting...")
            return "No CSV data provided"
        logger_flask.info("Flask: Start converting LiPD data to NOAA text")
        noaas = lpd_to_noaa(_json, "project", "1.0.0")
        logger_flask.info("Flask: Format the NOAA text data for the response object")
        for k,v in noaas.items():
            out.append({k: v})
        #logger_flask.info(noaas)
        logger_flask.info("Flask: Sending back NOAA files: {}".format(len(out)))
    except Exception as e:
        logger_flask.error("Flask App Error: {} : Quitting...".format(e))
        return "Exception found: {}".format(e)
    logger_flask.info("Flask: Sending Response to Node")
    return json.dumps(out)

@app.route('/getArchives', methods=['GET'])
@limiter.exempt
def get_archives():
    if not archives_map:
        return make_response(jsonify({'result': {}}), 200)
    covered = {'MarineSediment', 'LakeSediment', 'GlacierIce', 'GroundIce', 'TerrestrialSediment', 'MollusckShell','MolluskShell', 'MolluskShells', 'molluskshell'}
    final_arch = set()
    for key,value in archives_map.items():
        print(key,value)    
        if key in covered:
            continue
        elif key.title() in archives_map:
            final_arch.add(key.title())
        elif key.title() is not value:
            final_arch.add(key.title())
    return make_response(jsonify({'result': {'0' : list(sorted(final_arch))}}), 200)

@app.route('/predictNextValue', methods=['GET'])
@limiter.exempt
def predict_next_value():
    '''
    Method to predict the next value in the recommendation system chain.
    
    REMEMBER:
    1. If any component contains ',' it will cause a problem with the sentence being recommended. Replace the ',' with '' and then proceed to create the ',' separated sentence that will be fed to the API.
        example: archive = Marine Sediment & proxy observation type = D18O, sea water. Please replace the ',' in the proxy observation type before creating the input sentence for the API
        
    '''
    inputstr  = request.args.get('inputstr', None)
    inputstr = inputstr.replace('_','/')
    variabletype  = request.args.get('variableType', 'measured')
    inputs = inputstr.split(',')    
    variabletype = variabletype.strip().lower()

    if variabletype == 'measured' or variabletype == 'inferred':
        # HANDLE ARCHIVE TYPES USING EDIT DISTANCE FOR SPELLING MISTAKES
        if inputs[0] not in archives_map:
            present = False
            for key in archives_map.keys():
                dist = editDistDP(inputs[0].lower(), key.lower(), len(inputs[0]), len(key))
                if dist <= 3:
                    inputs[0] = archives_map[key]
                    present = True
                    break
            if not present:
                return make_response(jsonify({'result': {}}), 200)

        inputstr = (',').join(inputs)
        if inputs[0] in archives_for_MC:
            return predict_using_markov_chains(variabletype, inputstr)
        return predict_using_lstm(variabletype, inputstr)

    elif variabletype == 'time':
        if len(inputs) == 2 and inputs in set(time_map.keys()):
            return make_response(jsonify({'result': {'0': time_map[inputs[1]]}}), 200)
        return make_response(jsonify({'result': {'0' : list(time_map.keys())}}), 200)
    
    elif variabletype == 'depth':
        if len(inputs) == 2 and inputs[1] == 'Depth':
            return make_response(jsonify({'result': {'0' : ['m', 'cm', 'mm']}}), 200)
        return make_response(jsonify({'result': {'0' : ['Depth']}}), 200)
    
    else:
       return make_response(jsonify({'result': {}}), 200) 

@app.route('/autocomplete', methods=['GET'])
@limiter.limit("2/second", override_defaults=False)
def autocomplete_suggestion():
    
    global autocomplete_file_path

    # Ensure that autocomplete always works on the latest autocomplete data
    new_autocomplete_file_path = get_latest_file_with_path(flask_dir, 'autocomplete_file_*.json')
    if autocomplete_file_path != new_autocomplete_file_path:
        autocomplete_file_path = new_autocomplete_file_path
        load_names_set_from_file(new_autocomplete_file_path)


    fieldType  = request.args.get('fieldType', None)
    queryString  = request.args.get('queryString', '')
    results = []
    if fieldType not in names_set_ind_map:
        return make_response(jsonify({'result': {}}), 200) 
    if fieldType and queryString:
        queryString = queryString.lower()
        fieldType_set = names_set[fieldType]
        results.extend(py_.filter(fieldType_set, lambda word: word.lower().startswith(queryString)))
        # results.extend([word for word in fieldType_set if word.lower().startswith(queryString)])

        if len(queryString) >= avg_half_len_map[names_set_ind_map[fieldType]]:
            for word in fieldType_set:
                if word not in results:
                    edist = editDistDP(queryString, word.lower(), len(queryString), len(word))
                    if edist <= 5:
                        results.append(word)
    
    return make_response(jsonify({'result': {0: results}}), 200) 
        

def lpd_to_noaa(D, project, version, path=""):
    """
    Convert a LiPD format to NOAA format

    :param dict D: Metadata
    :param str project: Project Name
    :param float version: Project version
    :return dict D: Metadata
    """
    try:
        logger_flask.info("lpd_noaa: Get DSN")
        dsn = get_dsn(D)
        # Remove all the characters that are not allowed here. Since we're making URLs, they have to be compliant.
        logger_flask.info("lpd_noaa: Cleanup dsn, project, version with regexes")
        dsn = re.sub(r'[^A-Za-z-.0-9]', '', dsn)
        project = re.sub(r'[^A-Za-z-.0-9]', '', project)
        version = re.sub(r'[^A-Za-z-.0-9]', '', version)
        # Create the conversion object, and start the conversion process
        logger_flask.info("lpd_noaa: Create conversion object")
        _convert_obj = LPD_NOAA(D, dsn, project, version, path)
        logger_flask.info("lpd_noaa: Run conversion main()")
        _convert_obj.main()
        # get our new, modified master JSON from the conversion object
        # d = _convert_obj.get_master()
        logger_flask.info("lpd_noaa: Retrieve NOAA texts from conversion object")
        noaas = _convert_obj.get_noaa_texts()
        logger_flask.info("lpd_noaa: Log the NOAA texts below: ----------------")
        #logger_flask.info(noaas)
        # remove any root level urls that are deprecated
        # d = __rm_wdc_url(d)
    except Exception as e:
        logger_flask.error("lpd_to_noaa: {}".format(e))

    logger_flask.info("lpd_noaa: Exiting lpd_noaa")
    return noaas


def __rm_wdc_url(d):
    """
    Remove the WDCPaleoUrl key. It's no longer used but still exists in some files.
    :param dict d: Metadata
    :return dict d: Metadata
    """
    if "WDCPaleoUrl" in d:
        del d["WDCPaleoUrl"]
    if "WDSPaleoUrl" in d:
        del d["WDSPaleoUrl"]
    return d

def predict_using_markov_chains(variabletype, sentence):
    '''
    Method to return the list of top 5 values for a fieldType given the input sentence and the variableType using the model created for Markov Chains.

    Parameters
    ----------
    variabletype : string
        Either "measured" or "inferred".
    sentence : string
        Comma-separated input string containing values corresponding to the prediction chain.

    Returns
    -------
    response
        Json response to the API call predictNextValue.

    '''
    output = {}
    inputs = sentence.split(',')
    if len(inputs) == 2 and variabletype == 'measured':
        output = {0: pred3MC.predict_seq(sentence, isInferred=(True if variabletype=='inferred' else False))['0'], 1: pred4MC.predict_seq(sentence, isInferred=(True if variabletype =='inferred' else False))['0']}
    else:
        output = {0: pred4MC.predict_seq(sentence, isInferred=(True if variabletype =='inferred' else False))['0']}        
        
    return make_response(jsonify({'result': output}), 200) 

def predict_using_lstm(variabletype, sentence):
    '''
    Method to return the list of top 5 values for a fieldType given the input sentence and the variableType using the model created for LSTM.

    Parameters
    ----------
    variabletype : string
        Either "measured" or "inferred".
    sentence : string
        Comma-separated input string containing values corresponding to the prediction chain.

    Returns
    -------
    response
        Json response to the API call predictNextValue.

    '''
    
    inverse_ref_dict = {val:key for key,val in predLSTM.reference_dict.items()}
    inverse_ref_dict_u = {val:key for key,val in predLSTM.reference_dict_u.items()}
    
    output = {}
    inputs = sentence.split(',')
    if len(inputs) == 2 and variabletype == 'measured':
        result_list_units = predLSTM.predictForSentence(sentence, isInferred=(True if variabletype =='inferred' else False))['0']
        result_list = predLSTM.predictForSentence(sentence, isInferred=(True if variabletype=='inferred' else False))['1']
        result_list_units = [(inverse_ref_dict_u[val] if val in inverse_ref_dict_u else val) for val in result_list_units]
        result_list = [(inverse_ref_dict[val] if val in inverse_ref_dict else val) for val in result_list]
        output = {0: result_list_units, 1: result_list}
    else:
        result_list = predLSTM.predictForSentence(sentence, isInferred=(True if variabletype=='inferred' else False))['0']
        result_list = [(inverse_ref_dict[val] if val in inverse_ref_dict else val) for val in result_list]
        output = {0: result_list}        
        
    return make_response(jsonify({'result': output}), 200)


def editDistDP(str1, str2, m, n):
    '''
    Calculates the edit distance between str1 and str2.

    Parameters
    ----------
    str1 : string
        Input string 1.
    str2 : TYPE
        Input string 2.
    m : int
        len of string 1
    n : int
        len of string 2

    Returns
    -------
    int
        Edit distance value between str1 and str2.

    '''
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]
 
    for i in range(m + 1):
        for j in range(n + 1):
 
            if i == 0:
                dp[i][j] = j
 
            elif j == 0:
                dp[i][j] = i
 
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
 
            else:
                dp[i][j] = 1 + min(dp[i][j-1],  
                                   dp[i-1][j],  
                                   dp[i-1][j-1])
 
    return dp[m][n]

@app.errorhandler(429)
def ratelimit_handler(e):
    '''
    Method to return a json error response to the UI incase the rate of invoking the API is exceeded to more than 2/sec.

    Parameters
    ----------
    e : error
        Response code when the rate has been exceeded.

    Returns
    -------
    response
        Json response to UI when the rate limit has been exceeded.

    '''
    return make_response(jsonify(error="ratelimit exceeded %s" % e.description), 429)





    