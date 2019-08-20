#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 16:02:50 2019

@author: nmei
"""
import re
import os
from glob import glob
from shutil import copyfile,rmtree
from nipype.interfaces.fsl import ICA_AROMA
from nipype.interfaces import freesurfer,fsl
import pandas as pd
import numpy as np
from utils import (create_fsl_FEAT_workflow_func,
                   create_registration_workflow,
                   create_highpass_filter_workflow,
                   create_simple_struc2BOLD)
from time import sleep

data_dir = '../data/converted'
raw_data = [item for item in glob(os.path.join(data_dir,
                             '*',
                             '*',
                             '*BOLD*.nii.gz')) if \
                ('PA' not in item) and ('AP' not in item) and ('Local' not in item)]

for idx in range(len(raw_data)):
    picked_data = raw_data[idx]
    nipype_workflow_name = 'nipype_workflow'
    n_session = re.findall('sess\d+',picked_data)
    n_run = re.findall('Run-\d+_',picked_data)
    is_first = np.logical_and(len(n_run) > 0,len(n_session) > 0)
    sub_name = np.unique(re.findall(r'CSI\d',picked_data))[0]
    
    ## 3. highpass filter
    ICAed_file_name = os.path.abspath(os.path.join('/'.join(picked_data.split('/')[:-1]),
                                   'outputs',
                                   'func',
                                   'ICA_AROMA',
                                   'denoised_func_data_nonaggr.nii.gz'))
    
    # just have to be numbers only
    x_sub = int(re.findall('\d+',sub_name)[0])
    x_session = int(re.findall('\d+',n_session[0])[0])
    x_run = int(re.findall('\d+',n_run[0])[0])
    HP_freq = 60
    TR = 0.85
    output_dir = os.path.join('/'.join(picked_data.split('/')[:-1]),
                              'outputs',
                              'func',
                              'ICAed_filtered')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    highpass_workflow = create_highpass_filter_workflow(HP_freq = HP_freq, 
                                                        TR = TR,
                                workflow_name = f"highpass_{x_sub}_{x_session}_{x_run}")
    highpass_workflow.base_dir = 'hpf'
    highpass_workflow.write_graph(dotfilename = f"highpass_temp.dot")
    highpass_workflow.inputs.inputspec.ICAed_file = ICAed_file_name
    highpass_workflow.inputs.addmean.out_file = os.path.abspath(os.path.join(output_dir,
                                                                                     'filtered.nii.gz'))
    highpass_workflow.run()
    for log_file in glob(os.path.join(highpass_workflow.base_dir,"*","*","*","*","*",'report.rst')):
        log_name = log_file.split('/')[-5]
        copyfile(log_file,os.path.join(output_dir,'log_{}.rst'.format(log_name)))
    for folder in os.listdir(os.path.join(highpass_workflow.base_dir,
                                          f"highpass_{x_sub}_{x_session}_{x_run}")):
        try:
            rmtree(os.path.join(highpass_workflow.base_dir,
                                f"highpass_{x_sub}_{x_session}_{x_run}",
                                folder))
        except:
            print(folder)
