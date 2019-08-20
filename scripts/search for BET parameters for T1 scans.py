#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:37:13 2019

@author: nmei
"""

import os
from glob import glob
from nipype.interfaces import fsl
from nilearn.plotting import plot_anat
import numpy as np
from matplotlib import pyplot as plt

working_dir = '../data/converted/BOLD5000_Structural'
T1_scans = glob(os.path.join(working_dir,
                             "*",
                             "T1*.nii"))
fracs = np.arange(0.30,0.31,0.01)

def bet(in_file, frac = 0.40, robust = True):
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
    skullstrip = fsl.BET()
    in_file = in_file
    skullstrip.inputs.in_file = os.path.abspath(in_file)
    skullstrip.inputs.out_file = os.path.abspath(
                                in_file.replace('.nii',
                                                f'_{frac}_brain.nii.gz')
                                )
    skullstrip.inputs.frac = frac
    skullstrip.inputs.robust = robust
    return skullstrip

for working_data in T1_scans[2:]:
    for frac in fracs:
        skullstrip = bet(working_data,frac = round(frac,2),)
        print(skullstrip.cmdline)
        skullstrip.run()
        fig,ax = plt.subplots(figsize=(12,12))
        plot_anat(skullstrip.inputs.out_file,
                  title = f'frac = {frac:.2f}',
                  threshold = 0,
                  draw_cross = False,
    #              display_mode = 'z',
    #              cut_coords = np.arange(-40,41,5),
                  cut_coords = (0,0,0),
                  black_bg = True,
                  figure = fig,
                  axes = ax,)
        figure_dir = '/'.join(working_data.split('/')[:-1])
        fig.savefig(os.path.join(figure_dir,
                                 f"bet_{frac:.2f}.png"),
        dpi = 300,
        facecolor = 'k',
        edgecolor = 'k',)
        plt.close('all')