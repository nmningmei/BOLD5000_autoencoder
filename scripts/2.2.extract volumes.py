#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 11:33:47 2019

@author: nmei

extract volumes based on the time courses, without knowledge of the image

The following image presentation details apply for each run, each session, 
and each participant. A slow event-related design was implemented for stimulus 
presentation in order to isolate the blood oxygen level dependent (BOLD) signal 
for each individual image trial. At the beginning and end of each run, centered 
on a blank black screen, a fixation cross was shown for 6 sec and 12 sec, respectively. 
Following the initial fixation cross, all 37 stimuli were shown sequentially. 
Each image was presented for 1 sec followed by a 9 sec fixation cross. Given that 
each run contains 37 stimuli, there was a total of 370 sec of stimulus presentation 
plus fixation. Including the pre- and post-stimulus fixations, there were a 
total of 388 sec (6 min 28 sec) of data acquired in each run.

TR = 2 s
TE = 30 ms

Extract volumes 4 - 8 seconds after the onset of the image
"""

import os
import re
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from nilearn.input_data import NiftiMasker
from nipype.interfaces import afni,fsl
from nilearn.image import resample_img
from nibabel import load as load_fmri

data_dir    = '../data/converted'
reshaped    = glob(
                os.path.join(
                            data_dir,
                            "*/*/*/*/*",
                            "filtered_reshaped.nii.gz"))
target_func = load_fmri(os.path.abspath(os.path.join(data_dir,'target_func.nii.gz')))
saving_dir  = '../data/volume_of_interest'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)

for idx in tqdm(range(len(reshaped))):
    picked_data         = reshaped[idx]
    sub_name            = re.findall(r'CSI\d',picked_data)[0]
    n_session           = int(re.findall(r'\d+',re.findall(r'Sess-\d+_',picked_data)[0])[0])
    n_run               = int(re.findall(r'\d+',re.findall(r'Run-\d+',picked_data)[0])[0])
    picked_data_mask    = os.path.join('/'.join(picked_data.split('/')[:-2]),
                                       'mask.nii.gz')
    
    ###########################################################################
    resample3d = afni.utils.Resample(voxel_size = (2.386364,2.386364,2.4))
    resample3d.inputs.in_file = picked_data_mask
    resample3d.inputs.outputtype = 'NIFTI_GZ'
    resample3d.inputs.out_file = picked_data_mask.replace('mask.nii.gz',
                                                          'mask_resample.nii.gz')
    print(resample3d.cmdline)
    resample3d.run()
    
    resampled = resample_img(resample3d.inputs.out_file,
                             target_affine = target_func.affine,
                             target_shape = (88,88,66))
    resampled.to_filename(picked_data_mask.replace('mask.nii.gz',
                                                   'mask_reshaped.nii.gz'))
    binarize = fsl.ImageMaths(op_string = '-bin')
    binarize.inputs.in_file = picked_data_mask.replace('mask.nii.gz',
                                                       'mask_reshaped.nii.gz')
    binarize.inputs.out_file = picked_data_mask.replace('mask.nii.gz',
                                                        'mask_reshaped_bin.nii.gz')
    binarize.run()
    os.remove(picked_data_mask.replace('mask.nii.gz','mask_resample.nii.gz'))
    os.remove(picked_data_mask.replace('mask.nii.gz','mask_reshaped.nii.gz'))
    
    ###########################################################################
    masker              = NiftiMasker(mask_img      = picked_data_mask.replace('mask.nii.gz',
                                                                               'mask_reshaped_bin.nii.gz'),
#                                      standardize   = True,
#                                      detrend       = True,
#                                      t_r           = 2,
                                      )
    BOLD                = masker.fit_transform(picked_data)
    timepoints          = np.arange(start = 0,stop = 400,step = 2)[:BOLD.shape[0]]
    df                  = pd.DataFrame()
    df['timepoints']    = timepoints
    
    trial_start         = np.arange(start = 6,stop = timepoints.max() - 12,step = 10)
    interest_start      = trial_start + 4
    interest_stop       = trial_start + 8
    
    temp                = []
    for time in timepoints:
        if any([np.logical_and(interval[0] <= time,time <= interval[1]) for interval in zip(interest_start,interest_stop)]):
            temp.append(1)
        else:
            temp.append(0)
    df['volume_of_interest']    = temp
    idx_picked                  = list(df[df['volume_of_interest'] == 1].index)
    BOLD_picked                 = BOLD[idx_picked]
    
    for ii,sample in enumerate(BOLD_picked):
        back_to_3D  = masker.inverse_transform(sample)
        saving_name = os.path.join(saving_dir,
                                   f'{sub_name}_session{n_session}_run{n_run}_volume{ii+1}.nii.gz')
        back_to_3D.to_filename(saving_name)
        
        
