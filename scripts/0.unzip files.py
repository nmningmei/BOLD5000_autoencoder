#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 15:07:39 2019

@author: nmei
"""

import os
import zipfile
from glob import glob

working_dir = '../../../../BOLD5000/data/raw'
saving_dir = '../../../../BOLD5000/data/unzipped'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)

working_data = glob(os.path.join(working_dir,
                                 '*.zip'))
working_data = [item for item in working_data if \
                ('Unfiltered' in item) and \
                ('sess' in item)]

for f in working_data:
    file_name = f.split('/')[-1].replace('.zip','')
    with zipfile.ZipFile(f,'r') as zip_ref:
        if not os.path.exists(os.path.join(saving_dir,
                                           file_name)):
            os.mkdir(os.path.join(saving_dir,
                                  file_name))
        zip_ref.extractall(path = os.path.join(saving_dir,
                                               file_name))
