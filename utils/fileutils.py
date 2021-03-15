# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 12:54:49 2021

@author: shrav
"""

import os
import glob

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
    
