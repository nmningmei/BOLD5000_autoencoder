#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 10:53:56 2019

@author: nmei
"""

import os
from glob               import glob
from nipype.interfaces  import afni
from nilearn.image      import resample_img
from nibabel            import load as load_fmri

data_dir = '../data/converted'
filtered = glob(
        os.path.join(
                data_dir,
                "*/*/*/*/*",
                "filtered.nii.gz"))
target_func = load_fmri(os.path.abspath(os.path.join(data_dir,'target_func.nii.gz')))

for idx in range(len(filtered)):
    picked_data                     = os.path.abspath(filtered[idx])
    resample3d                      = afni.utils.Resample(voxel_size = (2.386364,2.386364,2.4))
    resample3d.inputs.in_file       = picked_data
    resample3d.inputs.outputtype    = 'NIFTI_GZ'
    resample3d.inputs.out_file      = picked_data.replace('filtered.nii.gz',
                                                          'filtered_resample.nii.gz')
    print(resample3d.cmdline)
    resample3d.run()

    resampled = resample_img(resample3d.inputs.out_file,
                             target_affine  = target_func.affine,
                             target_shape   = (88,88,66))
    resampled.to_filename(picked_data.replace('filtered.nii.gz',
                                              'filtered_reshaped.nii.gz'))
    # the existing resampled file will cause a conflict on the next run of this script
    # and we don't need it anymore
    os.remove(picked_data.replace('filtered.nii.gz','filtered_resample.nii.gz'))
