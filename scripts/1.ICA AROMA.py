#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:19:47 2019

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
    picked_data = os.path.abspath(raw_data[idx])
    ICAed_file_name = os.path.abspath(
            os.path.join('/'.join(picked_data.split('/')[:-1]),
                                   'outputs',
                                   'func',
                                   'ICA_AROMA',
                                   'denoised_func_data_nonaggr.nii.gz'))
    
    if not os.path.exists(ICAed_file_name):
        ## 2. ICA-AROMA
        sub_name = np.unique(re.findall(r'CSI\d',picked_data))[0]
        n_session = np.unique(re.findall(r'Sess-\d+',picked_data))[0]
        n_run = np.unique(re.findall(r'Run-\d+',picked_data))[0]
        session1 = 'Sess-1_'
        run1 = 'Run-1_'
        first_run = os.path.abspath([item for item in glob(os.path.join(data_dir,
                                      "*",
                                      "*",
                                      "*.nii.gz")) if \
                    (sub_name in item)\
                    and (session1 in item)\
                    and (run1 in item)][0])
        first_run_dir = '/'.join(first_run.split('/')[:-1])
        func_to_struct      = os.path.join(first_run_dir,
                                           'outputs',
                                           'reg',
                                           'example_func2highres.mat')
        warpfield           = os.path.join(first_run_dir,
                                           'outputs',
                                           'reg',
                                           'highres2standard_warp.nii.gz')
        fsl_mcflirt_movpar  = os.path.join(first_run_dir,
                                           'outputs',
                                           'func',
                                           'MC',
                                           'MCflirt.par')
        mask                = os.path.join(first_run_dir,
                                           'outputs',
                                           'func',
                                           'mask.nii.gz')
        output_dir          = os.path.join('/'.join(picked_data.split('/')[:-1]),
                                           'outputs',
                                           'func',
                                           'ICA_AROMA')
        preprocessed_fmri   = glob(os.path.join('/'.join(picked_data.split('/')[:-1]),
                                                'outputs',
                                                'func',
                                                'prefiltered_func.nii.gz'))[0]
        AROMA_obj           = ICA_AROMA()
        AROMA_obj.inputs.in_file            = os.path.abspath(preprocessed_fmri)
        AROMA_obj.inputs.mat_file           = os.path.abspath(func_to_struct)
        AROMA_obj.inputs.fnirt_warp_file    = os.path.abspath(warpfield)
        AROMA_obj.inputs.motion_parameters  = os.path.abspath(fsl_mcflirt_movpar)
        AROMA_obj.inputs.mask               = os.path.abspath(mask)
        AROMA_obj.inputs.denoise_type       = 'nonaggr'
        AROMA_obj.inputs.out_dir            = os.path.abspath(output_dir)
        cmdline             = 'python ../ICA_AROMA/' + AROMA_obj.cmdline + ' -ow'
        template = f"""
#!/bin/bash
#$ -cwd
#$ -o ../log/out_{sub_name}_{n_session}_{n_run}.txt
#$ -e ../log/err_{sub_name}_{n_session}_{n_run}.txt
#$ -m be
#$ -q fsl.q
#$ -M nmei@bcbl.eu
#$ -N "ICA|{sub_name}|{n_session}|{n_run}"
#$ -S /bin/bash

module load rocks-fsl-5.0.10
module load rocks-python-2.7
{cmdline}
        
        """
        f_name = f"ICA_{sub_name}_{n_session}_{n_run}"
        with open(f_name,'w') as f:
            f.write(template)
            f.close()
        
        os.system(f'qsub {f_name}')
    else:
        print(ICAed_file_name)