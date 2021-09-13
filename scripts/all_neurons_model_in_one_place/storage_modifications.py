# -*- coding: utf-8 -*-
"""
Created on Thu May  6 19:42:17 2021

@author: mohsen
"""

import os
import numpy as np


def last_ensemble_index_in_group(ensembles_path):
    exsiting_indices = [int(name) for name in os.listdir(ensembles_path) if os.path.isdir(os.path.join(ensembles_path,name))]
    
    if len(exsiting_indices) == 0:
        last_index = 0
    else:
        last_index = np.max(exsiting_indices)
    return last_index


def make_inner_dir(parent_dir, child_dir_name):
    path = os.path.join(parent_dir, child_dir_name)
    try:
        os.makedirs( path )
    except:
        pass
    return path

def save_field_period(dominant_mod_period, dominant_mod_intensity, ensemble_path):
    file_path = os.path.join(ensemble_path,'field_properties.txt')
    with open(file_path, "w") as file: 
        file.write( '{0},{1}'.format(dominant_mod_period, dominant_mod_intensity) )
    return

def save_sigma(reported_sigma,ensemble_path):
    file_path = os.path.join(ensemble_path,'sigma.txt')
    with open(file_path, "w") as file: 
        file.write( '{}'.format(reported_sigma) )
    return

def save_amin_saman_param(reported_param,ensemble_path):
    file_path = os.path.join(ensemble_path,'amin_saman_param.txt')
    with open(file_path, "w") as file: 
        file.write( '{}'.format(reported_param) )
    return

def save_mean_spiking_periods(mean_periods,ensemble_path):
    file_path = os.path.join(ensemble_path,'mean_spiking_periods.txt')
    np.savetxt(file_path,mean_periods,delimiter = ',')
    return