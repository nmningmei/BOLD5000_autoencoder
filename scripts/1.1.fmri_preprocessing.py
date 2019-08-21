#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 13:31:21 2019

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

    if not os.path.exists(os.path.join(
            '/'.join(picked_data.split('/')[:-1]),
            'outputs',
            'func',
            'prefiltered_func.nii.gz')):
        nipype_workflow_name = 'nipype_workflow'
        n_session = re.findall('Sess-1_',picked_data)
        n_run = re.findall('Run-1_',picked_data)
        is_first = np.logical_and(len(n_run) > 0,len(n_session) > 0)

        if not is_first:
            sub_name = np.unique(re.findall(r'CSI\d',picked_data))[0]
            session1 = 'Sess-1_'
            run1 = 'Run-1_'
            first_run = os.path.abspath([item for item in glob(os.path.join(data_dir,
                                          "*",
                                          "*",
                                          "*.nii.gz")) if \
                        (sub_name in item)\
                        and (session1 in item)\
                        and (run1 in item)][0])
        else:
            first_run = True

        # initialize the workflow and specify the hyperparameters
        # workflowname:     will create a folder that contain all the logs
        # first_run:        specified above
        # func_data_file:   the path to the functional data, .nii format
        # fwhm:             spatial smoothing size
        preproc,MC_dir,output_dir = create_fsl_FEAT_workflow_func(
                workflow_name       = nipype_workflow_name,
                first_run           = first_run,
                func_data_file      = os.path.abspath(picked_data),
                fwhm                = 3,
                )

        # make a figure of the workflow
        preproc.write_graph()
        # run the workflow
        res             = preproc.run()

        # moving MCflirt results to MC folder in output directory
        copyfile(glob(os.path.join(preproc.base_dir,
                                   nipype_workflow_name,
                                   'MCFlirt/mapflow/_MCFlirt0/',
                                   '*.par'))[0],
                 os.path.join(MC_dir,'MCflirt.par'))
        copyfile(glob(os.path.join(preproc.base_dir,
                                   nipype_workflow_name,
                                   'MCFlirt/mapflow/_MCFlirt0/',
                                   '*rot*'))[0],
                 os.path.join(MC_dir,'rot.png'))
        copyfile(glob(os.path.join(preproc.base_dir,
                                   nipype_workflow_name,
                                   'MCFlirt/mapflow/_MCFlirt0/',
                                   '*trans*'))[0],
                 os.path.join(MC_dir,'trans.png'))
        copyfile(glob(os.path.join(preproc.base_dir,
                                   nipype_workflow_name,
                                   'MCFlirt/mapflow/_MCFlirt0/',
                                   '*disp*'))[0],
                 os.path.join(MC_dir,'disp.png'))
        copyfile(glob(os.path.join(preproc.base_dir,
                                   nipype_workflow_name,
                                   'graph.png'))[0],
                 os.path.join(output_dir,'graph.png'))
        # copy mask, mean_func, and prefiltered_func
        copyfile(glob(os.path.join(preproc.base_dir,
                                   nipype_workflow_name,
                              'dilatemask',
                              '*','*',
                              '*.nii.gz'))[0],
                os.path.abspath(os.path.join(
                    output_dir,'mask.nii.gz')))
        copyfile(glob(os.path.join(preproc.base_dir,
                                   nipype_workflow_name,
                                   'meanscale',
                                   '*','*','*.nii.gz'))[0],
                 os.path.abspath(os.path.join(
                    output_dir,'prefiltered_func.nii.gz')))
        copyfile(glob(os.path.join(preproc.base_dir,
                                   nipype_workflow_name,
                                   'gen_mean_func_img',
                                   '*','*','*.nii.gz'))[0],
             os.path.abspath(os.path.join(
                    output_dir,'mean_func.nii.gz'))
            )
        for log_file in glob(os.path.join(preproc.base_dir,"*","*","*","*","*",'report.rst')):
            log_name = log_file.split('/')[-5]
            copyfile(log_file,os.path.join(output_dir,'log_{}.rst'.format(log_name)))
        # registration if only the first sesssion first run
        if first_run == True:
            # define the parent path of the structural scan and standard brain scans
            if os.path.exists('/opt/fsl-5.0.11/data/standard/MNI152_T1_2mm_brain.nii.gz'):
                standard_brain      = '/opt/fsl-5.0.11/data/standard/MNI152_T1_2mm_brain.nii.gz'
                standard_head       = '/opt/fsl-5.0.11/data/standard/MNI152_T1_2mm.nii.gz'
                standard_mask       = '/opt/fsl-5.0.11/data/standard/MNI152_T1_2mm_brain_mask_dil.nii.gz'
                fsl_bin             = '/opt/fsl-5.0.11/bin/'
            elif os.path.exists('/opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz'):
                standard_brain      = '/opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz'
                standard_head       = '/opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm.nii.gz'
                standard_mask       = '/opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm_brain_mask_dil.nii.gz'
                fsl_bin             = '/opt/fsl/fsl-5.0.10/fsl/bin/'
            # specify the path of the structural scan with and without the skull
            sub_name            = np.unique(re.findall(r'CSI\d',picked_data))[0]
            anat_brain          =  os.path.abspath(os.path.join(data_dir,
                                                    'BOLD5000_Structural',
                                                    f'{sub_name}_Structural',
                                                    f'T1w_MPRAGE_{sub_name}_0.3_brain.nii.gz'))# BET
            anat_head           = os.path.abspath(os.path.join(data_dir,
                                                    'BOLD5000_Structural',
                                                    f'{sub_name}_Structural',
                                                    f'T1w_MPRAGE_{sub_name}.nii'))
            # the so-called "example_func.nii.gz"
            func_ref            = os.path.join(preproc.base_dir,
                                                     'outputs',
                                                     'func',
                                                     'example_func.nii.gz')
            # define the output path for saving the coregistration results
            output_dir          = os.path.join(preproc.base_dir,
                                               'outputs',
                                               'reg')
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            # create the registration workflow
            # anat_brain    : path of the structural scan after BET
            # anat_head     : path of the structural scan before BET
            # func_ref      : the so-called "example_func.nii.gz'
            # standard_brain: MNI brain after BET
            # standard_head : MNI brain before BET
            # standard_mask : mask of BET for MNI brain
            registration        = create_registration_workflow(
                                        anat_brain,
                                        anat_head,
                                        func_ref,
                                        standard_brain,
                                        standard_head,
                                        standard_mask,
                                        workflow_name = 'registration',
                                        output_dir = output_dir)
            registration.write_graph()
            registration.run()


            ######################
            ###### plotting ######
            example_func2highres = os.path.abspath(os.path.join(output_dir,
                                                                'example_func2highres'))
            example_func2standard = os.path.abspath(os.path.join(output_dir,
                                                                 "example_func2standard"))
            highres2standard = os.path.abspath(os.path.join(output_dir,
                                                            'highres2standard'))
            highres = os.path.abspath(anat_brain)
            standard = os.path.abspath(standard_brain)

            plot_example_func2highres = f"""
{fsl_bin}slicer {example_func2highres} {highres} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ;
{fsl_bin}pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2highres}1.png ;
{fsl_bin}slicer {highres} {example_func2highres} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ;
{fsl_bin}pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2highres}2.png ;
{fsl_bin}pngappend {example_func2highres}1.png - {example_func2highres}2.png {example_func2highres}.png;
/bin/rm -f sl?.png {example_func2highres}2.png
/bin/rm {example_func2highres}1.png
            """.replace("\n"," ")

            plot_highres2standard = f"""
{fsl_bin}slicer {highres2standard} {standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ;
{fsl_bin}pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {highres2standard}1.png ;
{fsl_bin}slicer {standard} {highres2standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ;
{fsl_bin}pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {highres2standard}2.png ;
{fsl_bin}pngappend {highres2standard}1.png - {highres2standard}2.png {highres2standard}.png;
/bin/rm -f sl?.png {highres2standard}2.png
/bin/rm {highres2standard}1.png
            """.replace("\n"," ")

            plot_example_func2standard = f"""
{fsl_bin}slicer {example_func2standard} {standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ;
{fsl_bin}pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2standard}1.png ;
{fsl_bin}slicer {standard} {example_func2standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ;
{fsl_bin}pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2standard}2.png ;
{fsl_bin}pngappend {example_func2standard}1.png - {example_func2standard}2.png {example_func2standard}.png;
/bin/rm -f sl?.png {example_func2standard}2.png
        """.replace("\n"," ")
            for cmdline in [plot_example_func2highres,plot_example_func2standard,plot_highres2standard]:
                os.system(cmdline)


        # remove the logs
        rmtree(os.path.join(preproc.base_dir,
                            nipype_workflow_name))
