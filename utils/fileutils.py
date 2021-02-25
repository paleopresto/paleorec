# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 12:54:49 2021

@author: shrav
"""

import os
import glob

def get_latest_file_with_path(path, *paths):
    fullpath = os.path.join(path, *paths)
    list_of_files = glob.iglob(fullpath)  
    if not list_of_files:                
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file
    
