#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 13:25:01 2019

@author: nmei
"""

import os
from glob import glob
from tqdm import tqdm
from shutil import copyfile
from sklearn.model_selection import train_test_split

data_dir = '../data/volume_of_interest/'
train_dir = '../data/train/'
validation_dir = '../data/validation/'
for d in [train_dir,validation_dir]:
    if not os.path.exists(d):
        os.mkdir(d)

all_files = glob(os.path.join(data_dir,"*.nii.gz"))

train,test = train_test_split(all_files,test_size = 0.2,random_state = 12345)

for f in tqdm(train):
    copyfile(f,os.path.join(train_dir,f.split('/')[-1]))
    
for f in tqdm(test):
    copyfile(f,os.path.join(validation_dir,f.split('/')[-1]))