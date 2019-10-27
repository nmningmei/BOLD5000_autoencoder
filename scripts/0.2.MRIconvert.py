#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:33:29 2019

@author: nmei

While dcm2niix is provided with recent versions of MRIcroGL and MRIcron, it can also be installed on its own and 
does not require those tools. Methods to get the latest stable release include
dcm2niix GitHub release page.
- For Linux: curl -fLO https://github.com/rordenlab/dcm2niix/releases/latest/download/dcm2niix_lnx.zip
- For MacOS: curl -fLO https://github.com/rordenlab/dcm2niix/releases/latest/download/dcm2niix_mac.zip
- For Windows: curl -fLO https://github.com/rordenlab/dcm2niix/releases/latest/download/dcm2niix_win.zip


With the current versions of dcm2niix you can disable rotation of 3D acquisitions with the -x i parameter. 
Note that only 3D acquisitions are rotated (as these scans do not require slice time correction that is sometimes 
applied to 2D EPI sequences) and is lossless: the volume is rotated orthogonally to the orientation that best matches 
the NIfTI identity matrix. The residual rotation is stored in the Form so spatial positions remain the same. Therefore, 
this only influence data storage on disk. Further, as 2D EPI scans are typically axial slices while 3D sequences of 
the head are usually sagittal, this rotation (which makes stores the 3D image as axial slices) makes the 2D and 3D 
sequences more similar.
"""

import os
from glob import glob
from nipype.interfaces.dcm2nii import Dcm2nii,Dcm2niix
import numpy as np


output_dir = f'../data/converted'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


working_dir = '../../../../BOLD5000/data/unzipped'# data in .dcm format
working_data = [item for item in glob(os.path.join(working_dir,'*','*'))]
print(np.sort(working_data),'\n')

#structural = [item for item in glob(os.path.join(working_dir,'*')) if \
#             ('t1' in item)][-1]

for working_file in working_data:
    converter = Dcm2niix()
    converter.inputs.source_dir = os.path.abspath(working_file)
    temp_output_dir = os.path.abspath(os.path.join(
        output_dir,'/'.join(working_file.split('/')[-2:]),
                                    ))
    if not os.path.exists(temp_output_dir):
        os.makedirs(temp_output_dir)
    converter.inputs.output_dir = temp_output_dir
    converter.inputs.bids_format = True
    converter.inputs.single_file = True
    converter.inputs.crop = False
    converter.cmdline
    converter.run()

#for working_file in [structural]:
#    converter = Dcm2nii()
#    converter.inputs.source_dir = os.path.abspath(working_file)
#    temp_output_dir = os.path.abspath(os.path.join(
#        output_dir,'/'.join(working_file.split('/')[-3:]),
#                                    ))
#    if not os.path.exists(temp_output_dir):
#        os.makedirs(temp_output_dir)
#    converter.inputs.output_dir = temp_output_dir
#    converter.inputs.gzip_output = True
#    converter.inputs.reorient_and_crop = False
#    converter.run()
