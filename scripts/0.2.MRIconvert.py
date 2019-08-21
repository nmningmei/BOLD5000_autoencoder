#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:33:29 2019

@author: nmei

dcm2niix will re-orient the image automatically and cannot be turned off
so I will stick to dcm2nii for a while and see if anything changes
** but after a throughout investigation, this would only affect structural scans

dcm2nii requires mricron-10.2014 while dcm2niix requires mricrogl
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

#structual = [item for item in glob(os.path.join(working_dir,'*')) if \
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

#for working_file in [structual]:
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
