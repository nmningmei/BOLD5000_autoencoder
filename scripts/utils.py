#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 15:05:36 2019

@author: nmei
"""

from autoreject import (AutoReject,get_rejection_threshold)
import mne
from glob import glob
import re
import os

import numpy as np
import pandas as pd
import pickle

from sklearn.metrics                               import roc_auc_score,roc_curve
from sklearn.metrics                               import (
                                                           classification_report,
                                                           matthews_corrcoef,
                                                           confusion_matrix,
                                                           f1_score,
                                                           log_loss,
                                                           r2_score
                                                           )

from sklearn.preprocessing                         import (MinMaxScaler,
                                                           OneHotEncoder,
                                                           FunctionTransformer,
                                                           StandardScaler)

from sklearn.pipeline                              import make_pipeline
from sklearn.ensemble.forest                       import _generate_unsampled_indices
from sklearn.utils                                 import shuffle
from sklearn.svm                                   import SVC,LinearSVC
from sklearn.calibration                           import CalibratedClassifierCV
from sklearn.decomposition                         import PCA
from sklearn.dummy                                 import DummyClassifier
from sklearn.feature_selection                     import (SelectFromModel,
                                                           SelectPercentile,
                                                           VarianceThreshold,
                                                           mutual_info_classif,
                                                           f_classif,
                                                           chi2,
                                                           f_regression,
                                                           GenericUnivariateSelect)
from sklearn.model_selection                       import (StratifiedShuffleSplit,
                                                           cross_val_score)
from sklearn.ensemble                              import RandomForestClassifier,BaggingClassifier,VotingClassifier
from sklearn.neural_network                        import MLPClassifier
from xgboost                                       import XGBClassifier
from itertools                                     import product,combinations
from sklearn.base                                  import clone
from sklearn.neighbors                             import KNeighborsClassifier
from sklearn.tree                                  import DecisionTreeClassifier
from collections                                   import OrderedDict

from scipy                                         import stats
from collections                                   import Counter

import matplotlib.pyplot  as plt
import matplotlib.patches as patches

try:
    #from mvpa2.datasets.base                           import Dataset
    from mvpa2.mappers.fx                              import mean_group_sample
    #from mvpa2.measures                                import rsa
    #from mvpa2.measures.searchlight                    import sphere_searchlight
    #from mvpa2.base.learner                            import ChainLearner
    #from mvpa2.mappers.shape                           import TransposeMapper
    #from mvpa2.generators.partition                    import NFoldPartitioner
except:
    pass#print('pymvpa is not installed')
try:
#    from tqdm import tqdm_notebook as tqdm
    from tqdm.auto import tqdm
except:
    print('why is tqdm not installed?')
    
    

def get_brightness_threshold(thresh):
    return [0.75 * val for val in thresh]

def get_brightness_threshold_double(thresh):
    return [2 * 0.75 * val for val in thresh]

def cartesian_product(fwhms, in_files, usans, btthresh):
    from nipype.utils.filemanip import ensure_list
    # ensure all inputs are lists
    in_files                = ensure_list(in_files)
    fwhms                   = [fwhms] if isinstance(fwhms, (int, float)) else fwhms
    # create cartesian product lists (s_<name> = single element of list)
    cart_in_file            = [
            s_in_file for s_in_file in in_files for s_fwhm in fwhms
                                ]
    cart_fwhm               = [
            s_fwhm for s_in_file in in_files for s_fwhm in fwhms
                                ]
    cart_usans              = [
            s_usans for s_usans in usans for s_fwhm in fwhms
                                ]
    cart_btthresh           = [
            s_btthresh for s_btthresh in btthresh for s_fwhm in fwhms
                                ]
    return cart_in_file, cart_fwhm, cart_usans, cart_btthresh

def getusans(x):
    return [[tuple([val[0], 0.5 * val[1]])] for val in x]

def create_fsl_FEAT_workflow_func(whichrun          = 0,
                                  whichvol          = 'middle',
                                  workflow_name     = 'nipype_mimic_FEAT',
                                  first_run         = True,
                                  func_data_file    = 'temp',
                                  fwhm              = 3):
    from nipype.workflows.fmri.fsl             import preprocess
    from nipype.interfaces                     import fsl
    from nipype.interfaces                     import utility as util
    from nipype.pipeline                       import engine as pe
    """
    Setup some functions and hyperparameters
    """
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
    pickrun             = preprocess.pickrun
    pickvol             = preprocess.pickvol
    getthreshop         = preprocess.getthreshop
    getmeanscale        = preprocess.getmeanscale
#    chooseindex         = preprocess.chooseindex
    
    """
    Start constructing the workflow graph
    """
    preproc             = pe.Workflow(name = workflow_name)
    """
    Initialize the input and output spaces
    """
    inputnode           = pe.Node(
                        interface   = util.IdentityInterface(fields = ['func',
                                                                       'fwhm',
                                                                       'anat']),
                        name        = 'inputspec')
    outputnode          = pe.Node(
                        interface   = util.IdentityInterface(fields = ['reference',
                                                                       'motion_parameters',
                                                                       'realigned_files',
                                                                       'motion_plots',
                                                                       'mask',
                                                                       'smoothed_files',
                                                                       'mean']),
                        name        = 'outputspec')
    """
    first step: convert Images to float values
    """
    img2float           = pe.MapNode(
                        interface   = fsl.ImageMaths(
                                        out_data_type   = 'float',
                                        op_string       = '',
                                        suffix          = '_dtype'),
                        iterfield   = ['in_file'],
                        name        = 'img2float')
    preproc.connect(inputnode,'func',
                    img2float,'in_file')
    """
    delete first 10 volumes
    """
    develVolume         = pe.MapNode(
                        interface   = fsl.ExtractROI(t_min  = 0,
                                                     t_size = -1),
                        iterfield   = ['in_file'],
                        name        = 'remove_volumes')
    preproc.connect(img2float,      'out_file',
                    develVolume,    'in_file')
    if first_run == True:
        """ 
        extract example fMRI volume: middle one
        """
        extract_ref     = pe.MapNode(
                        interface   = fsl.ExtractROI(t_size = 1,),
                        iterfield   = ['in_file'],
                        name        = 'extractref')
        # connect to the deleteVolume node to get the data
        preproc.connect(develVolume,'roi_file',
                        extract_ref,'in_file')
        # connect to the deleteVolume node again to perform the extraction
        preproc.connect(develVolume,('roi_file',pickvol,0,whichvol),
                        extract_ref,'t_min')
        # connect to the output node to save the reference volume
        preproc.connect(extract_ref,'roi_file',
                        outputnode, 'reference')
    if first_run == True:
        """
        Realign the functional runs to the reference (`whichvol` volume of first run)
        """
        motion_correct  = pe.MapNode(
                        interface   = fsl.MCFLIRT(save_mats     = True,
                                                  save_plots    = True,
                                                  save_rms      = True,
                                                  stats_imgs    = True,
                                                  interpolation = 'spline',
                                                  output_type   = 'NIFTI_GZ'),
                        iterfield   = ['in_file','ref_file'],
                        name        = 'MCFlirt',
                                                  )
        # connect to the develVolume node to get the input data
        preproc.connect(develVolume,    'roi_file',
                        motion_correct, 'in_file',)
        ######################################################################################
        #################  the part where we replace the actual reference image if exists ####
        ######################################################################################
        # connect to the develVolume node to get the reference
        preproc.connect(extract_ref,    'roi_file', 
                        motion_correct, 'ref_file')
        ######################################################################################
        # connect to the output node to save the motion correction parameters
        preproc.connect(motion_correct, 'par_file',
                        outputnode,     'motion_parameters')
        # connect to the output node to save the other files
        preproc.connect(motion_correct, 'out_file',
                        outputnode,     'realigned_files')
    else:
        """
        Realign the functional runs to the reference (`whichvol` volume of first run)
        """
        motion_correct      = pe.MapNode(
                            interface   = fsl.MCFLIRT(ref_file      = first_run,
                                                      save_mats     = True,
                                                      save_plots    = True,
                                                      save_rms      = True,
                                                      stats_imgs    = True,
                                                      interpolation = 'spline',
                                                      output_type   = 'NIFTI_GZ'),
                            iterfield   = ['in_file','ref_file'],
                            name        = 'MCFlirt',
                        )
        # connect to the develVolume node to get the input data
        preproc.connect(develVolume,    'roi_file',
                        motion_correct, 'in_file',)
        # connect to the output node to save the motion correction parameters
        preproc.connect(motion_correct, 'par_file',
                        outputnode,     'motion_parameters')
        # connect to the output node to save the other files
        preproc.connect(motion_correct, 'out_file',
                        outputnode,     'realigned_files')
    """
    plot the estimated motion parameters
    """
    plot_motion             = pe.MapNode(
                            interface   = fsl.PlotMotionParams(in_source = 'fsl'),
                            iterfield   = ['in_file'],
                            name        = 'plot_motion',
            )
    plot_motion.iterables = ('plot_type',['rotations',
                                          'translations',
                                          'displacement'])
    preproc.connect(motion_correct, 'par_file',
                    plot_motion,    'in_file')
    preproc.connect(plot_motion,    'out_file',
                    outputnode,     'motion_plots')
    """
    extract the mean volume of the first functional run
    """
    meanfunc                = pe.Node(
                            interface  = fsl.ImageMaths(op_string   = '-Tmean',
                                                        suffix      = '_mean',),
                            name        = 'meanfunc')
    preproc.connect(motion_correct, ('out_file',pickrun,whichrun),
                    meanfunc,       'in_file')
    """
    strip the skull from the mean functional to generate a mask
    """
    meanfuncmask            = pe.Node(
                            interface   = fsl.BET(mask        = True,
                                                  no_output   = True,
                                                  frac        = 0.3,
                                                  surfaces    = True,),
                            name        = 'bet2_mean_func')
    preproc.connect(meanfunc,       'out_file',
                    meanfuncmask,   'in_file')
    """
    Mask the motion corrected functional data with the mask to create the masked (bet) motion corrected functional data
    """
    maskfunc                = pe.MapNode(
                            interface   = fsl.ImageMaths(suffix = '_bet',
                                                         op_string = '-mas'),
                            iterfield   = ['in_file'],
                            name        = 'maskfunc')
    preproc.connect(motion_correct, 'out_file',
                    maskfunc,       'in_file')
    preproc.connect(meanfuncmask,   'mask_file',
                    maskfunc,       'in_file2')
    """
    determine the 2nd and 98th percentiles of each functional run
    """
    getthreshold            = pe.MapNode(
                            interface   = fsl.ImageStats(op_string = '-p 2 -p 98'),
                            iterfield   = ['in_file'],
                            name        = 'getthreshold')
    preproc.connect(maskfunc,       'out_file',
                    getthreshold,   'in_file')
    """
    threshold the functional data at 10% of the 98th percentile
    """
    threshold               = pe.MapNode(
                            interface   = fsl.ImageMaths(out_data_type  = 'char',
                                                         suffix         = '_thresh',
                                                         op_string      = '-Tmin -bin'),
                            iterfield   = ['in_file','op_string'],
                            name        = 'tresholding')
    preproc.connect(maskfunc, 'out_file',
                    threshold,'in_file')
    """
    define a function to get 10% of the intensity
    """
    preproc.connect(getthreshold,('out_stat',getthreshop),
                    threshold,    'op_string')
    """
    Determine the median value of the functional runs using the mask
    """
    medianval               = pe.MapNode(
                            interface   = fsl.ImageStats(op_string = '-k %s -p 50'),
                            iterfield   = ['in_file','mask_file'],
                            name        = 'cal_intensity_scale_factor')
    preproc.connect(motion_correct,     'out_file',
                    medianval,          'in_file')
    preproc.connect(threshold,          'out_file',
                    medianval,          'mask_file')
    """
    dilate the mask
    """
    dilatemask              = pe.MapNode(
                            interface   = fsl.ImageMaths(suffix = '_dil',
                                                         op_string = '-dilF'),
                            iterfield   = ['in_file'],
                            name        = 'dilatemask')
    preproc.connect(threshold,  'out_file',
                    dilatemask, 'in_file')
    preproc.connect(dilatemask, 'out_file',
                    outputnode, 'mask')
    """
    mask the motion corrected functional runs with the dilated mask
    """
    dilateMask_MCed         = pe.MapNode(
                            interface   = fsl.ImageMaths(suffix     = '_mask',
                                                         op_string  = '-mas'),
                            iterfield   = ['in_file','in_file2'],
                            name        = 'dilateMask_MCed')
    preproc.connect(motion_correct,     'out_file',
                    dilateMask_MCed,    'in_file',)
    preproc.connect(dilatemask,         'out_file',
                    dilateMask_MCed,    'in_file2')
    """
    We now take this functional data that is motion corrected, high pass filtered, and
    create a "mean_func" image that is the mean across time (Tmean)
    """
    meanfunc2               = pe.MapNode(
                            interface   = fsl.ImageMaths(suffix     = '_mean',
                                                         op_string  = '-Tmean',),
                            iterfield   = ['in_file'],
                            name        = 'meanfunc2')
    preproc.connect(dilateMask_MCed,    'out_file',
                    meanfunc2,          'in_file')
    """
    smooth each run using SUSAN with the brightness threshold set to 
    75% of the median value for each run and a mask constituing the 
    mean functional
    """
    merge                   = pe.Node(
                            interface   = util.Merge(2, axis = 'hstack'), 
                            name        = 'merge')
    preproc.connect(meanfunc2,  'out_file', 
                    merge,      'in1')
    preproc.connect(medianval,('out_stat',get_brightness_threshold_double), 
                    merge,      'in2')
    smooth                  = pe.MapNode(
                            interface   = fsl.SUSAN(dimension   = 3,
                                                    use_median  = True),
                            iterfield   = ['in_file',
                                           'brightness_threshold',
                                           'fwhm',
                                           'usans'],
                            name        = 'susan_smooth')
    preproc.connect(dilateMask_MCed,    'out_file', 
                    smooth,             'in_file')
    preproc.connect(medianval,         ('out_stat',get_brightness_threshold),
                    smooth,             'brightness_threshold')
    preproc.connect(inputnode,          'fwhm', 
                    smooth,             'fwhm')
    preproc.connect(merge,              ('out',getusans),
                    smooth,             'usans')
    """
    mask the smoothed data with the dilated mask
    """
    maskfunc3               = pe.MapNode(
                            interface   = fsl.ImageMaths(suffix     = '_mask',
                                                         op_string  = '-mas'),
                            iterfield   = ['in_file','in_file2'],
                            name        = 'dilateMask_smoothed')
    # connect the output of the susam smooth component to the maskfunc3 node
    preproc.connect(smooth,     'smoothed_file',
                    maskfunc3,  'in_file')
    # connect the output of the dilated mask to the maskfunc3 node
    preproc.connect(dilatemask, 'out_file',
                    maskfunc3,  'in_file2')
    """
    scale the median value of the run is set to 10000
    """
    meanscale               = pe.MapNode(
                            interface   = fsl.ImageMaths(suffix = '_intnorm'),
                            iterfield   = ['in_file','op_string'],
                            name        = 'meanscale')
    preproc.connect(maskfunc3, 'out_file',
                    meanscale, 'in_file')
    preproc.connect(meanscale, 'out_file',
                    outputnode,'smoothed_files')
    """
    define a function to get the scaling factor for intensity normalization
    """
    preproc.connect(medianval,('out_stat',getmeanscale),
                    meanscale,'op_string')
    """
    generate a mean functional image from the first run
    should this be the 'mean.nii.gz' we will use in the future?
    """
    meanfunc3               = pe.MapNode(
                            interface   = fsl.ImageMaths(suffix     = '_mean',
                                                         op_string  = '-Tmean',),
                            iterfield   = ['in_file'],
                            name        = 'gen_mean_func_img')
    preproc.connect(meanscale, 'out_file',
                    meanfunc3, 'in_file')
    preproc.connect(meanfunc3, 'out_file',
                    outputnode,'mean')
    
    
    # initialize some of the input files
    preproc.inputs.inputspec.func       = os.path.abspath(func_data_file)
    preproc.inputs.inputspec.fwhm       = 3
    preproc.base_dir                    = os.path.abspath('/'.join(
                                        func_data_file.split('/')[:-1]))
    
    output_dir                          = os.path.abspath(os.path.join(
                                        preproc.base_dir,
                                        'outputs',
                                        'func'))
    MC_dir                              = os.path.join(output_dir,'MC')
    for directories in [output_dir,MC_dir]:
        if not os.path.exists(directories):
            os.makedirs(directories)
    if first_run == True:
        preproc.inputs.extractref.roi_file =\
                 os.path.abspath(os.path.join(
                             output_dir,
                             'example_func.nii.gz'))

    
    return preproc,MC_dir,output_dir

def create_registration_workflow(
                                 anat_brain,
                                 anat_head,
                                 example_func,
                                 standard_brain,
                                 standard_head,
                                 standard_mask,
                                 workflow_name = 'registration',
                                 output_dir = 'temp'):
    from nipype.interfaces          import fsl
    from nipype.interfaces         import utility as util
    from nipype.pipeline           import engine as pe
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
    registration                    = pe.Workflow(name = 'registration')
    inputnode                       = pe.Node(
                                        interface   = util.IdentityInterface(
                                        fields      = [
                                                'highres', # anat_brain
                                                'highres_head', # anat_head
                                                'example_func',
                                                'standard', # standard_brain
                                                'standard_head',
                                                'standard_mask'
                                                ]),
                                        name        = 'inputspec')
    outputnode                      = pe.Node(
                                    interface   = util.IdentityInterface(
                                    fields      = ['example_func2highres_nii_gz',
                                                   'example_func2highres_mat',
                                                   'linear_example_func2highres_log',
                                                   'highres2example_func_mat',
                                                   'highres2standard_linear_nii_gz',
                                                   'highres2standard_mat',
                                                   'linear_highres2standard_log',
                                                   'highres2standard_nii_gz',
                                                   'highres2standard_warp_nii_gz',
                                                   'highres2standard_head_nii_gz',
    #                                               'highres2standard_apply_warp_nii_gz',
                                                   'highres2highres_jac_nii_gz',
                                                   'nonlinear_highres2standard_log',
                                                   'highres2standard_nii_gz',
                                                   'standard2highres_mat',
                                                   'example_func2standard_mat',
                                                   'example_func2standard_warp_nii_gz',
                                                   'example_func2standard_nii_gz',
                                                   'standard2example_func_mat',
                                                   ]),
                                    name        = 'outputspec')
    """
    fslmaths /bcbl/home/public/Consciousness/uncon_feat/data/MRI/sub-01/anat/sub-01-T1W_mprage_sag_p2_1iso_MGH_day_6_nipy_brain highres
    fslmaths /bcbl/home/public/Consciousness/uncon_feat/data/MRI/sub-01/anat/sub-01-T1W_mprage_sag_p2_1iso_MGH_day_6_nipy_brain  highres_head
    fslmaths /opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm_brain standard
    fslmaths /opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm standard_head
    fslmaths /opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm_brain_mask_dil standard_mask
    """
    # skip
    
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/flirt 
        -in example_func 
        -ref highres 
        -out example_func2highres 
        -omat example_func2highres.mat 
        -cost corratio 
        -dof 7 
        -searchrx -180 180 
        -searchry -180 180 
        -searchrz -180 180 
        -interp trilinear 
    """
    linear_example_func2highres = pe.MapNode(
            interface   = fsl.FLIRT(cost = 'corratio',
                                    interp = 'trilinear',
                                    dof = 7,
                                    save_log = True,
                                    searchr_x = [-180, 180],
                                    searchr_y = [-180, 180],
                                    searchr_z = [-180, 180],),
            iterfield   = ['in_file','reference'],
            name        = 'linear_example_func2highres')
    registration.connect(inputnode, 'example_func',
                         linear_example_func2highres, 'in_file')
    registration.connect(inputnode, 'highres',
                         linear_example_func2highres, 'reference')
    registration.connect(linear_example_func2highres, 'out_file',
                         outputnode, 'example_func2highres_nii_gz')
    registration.connect(linear_example_func2highres, 'out_matrix_file',
                         outputnode, 'example_func2highres_mat')
    registration.connect(linear_example_func2highres, 'out_log',
                         outputnode, 'linear_example_func2highres_log')
    
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
        -inverse -omat highres2example_func.mat example_func2highres.mat
    """
    get_highres2example_func = pe.MapNode(
            interface = fsl.ConvertXFM(invert_xfm = True),
            iterfield = ['in_file'],
            name = 'get_highres2example_func')
    registration.connect(linear_example_func2highres,'out_matrix_file',
                         get_highres2example_func,'in_file')
    registration.connect(get_highres2example_func,'out_file',
                         outputnode,'highres2example_func_mat')
    
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/flirt 
        -in highres 
        -ref standard 
        -out highres2standard 
        -omat highres2standard.mat 
        -cost corratio 
        -dof 12 
        -searchrx -180 180 
        -searchry -180 180 
        -searchrz -180 180 
        -interp trilinear 
    """
    linear_highres2standard = pe.MapNode(
            interface = fsl.FLIRT(cost = 'corratio',
                                interp = 'trilinear',
                                dof = 12,
                                save_log = True,
                                searchr_x = [-180, 180],
                                searchr_y = [-180, 180],
                                searchr_z = [-180, 180],),
            iterfield = ['in_file','reference'],
            name = 'linear_highres2standard')
    registration.connect(inputnode,'highres',
                         linear_highres2standard,'in_file')
    registration.connect(inputnode,'standard',
                         linear_highres2standard,'reference',)
    registration.connect(linear_highres2standard,'out_file',
                         outputnode,'highres2standard_linear_nii_gz')
    registration.connect(linear_highres2standard,'out_matrix_file',
                         outputnode,'highres2standard_mat')
    registration.connect(linear_highres2standard,'out_log',
                         outputnode,'linear_highres2standard_log')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/fnirt 
        --iout=highres2standard_head 
        --in=highres_head 
        --aff=highres2standard.mat 
        --cout=highres2standard_warp 
        --iout=highres2standard 
        --jout=highres2highres_jac 
        --config=T1_2_MNI152_2mm 
        --ref=standard_head 
        --refmask=standard_mask 
        --warpres=10,10,10
    """
    nonlinear_highres2standard = pe.MapNode(
            interface = fsl.FNIRT(warp_resolution = (10,10,10),
                                  config_file = "T1_2_MNI152_2mm"),
            iterfield = ['in_file','ref_file','affine_file','refmask_file'],
            name = 'nonlinear_highres2standard')
    # -- iout
    registration.connect(nonlinear_highres2standard,'warped_file',
                         outputnode,'highres2standard_head_nii_gz')
    # --in
    registration.connect(inputnode,'highres',
                         nonlinear_highres2standard,'in_file')
    # --aff
    registration.connect(linear_highres2standard,'out_matrix_file',
                         nonlinear_highres2standard,'affine_file')
    # --cout
    registration.connect(nonlinear_highres2standard,'fieldcoeff_file',
                         outputnode,'highres2standard_warp_nii_gz')
    # --jout
    registration.connect(nonlinear_highres2standard,'jacobian_file',
                         outputnode,'highres2highres_jac_nii_gz')
    # --ref
    registration.connect(inputnode,'standard_head',
                         nonlinear_highres2standard,'ref_file',)
    # --refmask
    registration.connect(inputnode,'standard_mask',
                         nonlinear_highres2standard,'refmask_file')
    # log
    registration.connect(nonlinear_highres2standard,'log_file',
                         outputnode,'nonlinear_highres2standard_log')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/applywarp 
        -i highres 
        -r standard 
        -o highres2standard 
        -w highres2standard_warp
    """
    warp_highres2standard = pe.MapNode(
            interface = fsl.ApplyWarp(),
            iterfield = ['in_file','ref_file','field_file'],
            name = 'warp_highres2standard')
    registration.connect(inputnode,'highres',
                         warp_highres2standard,'in_file')
    registration.connect(inputnode,'standard',
                         warp_highres2standard,'ref_file')
    registration.connect(warp_highres2standard,'out_file',
                         outputnode,'highres2standard_nii_gz')
    registration.connect(nonlinear_highres2standard,'fieldcoeff_file',
                         warp_highres2standard,'field_file')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
        -inverse -omat standard2highres.mat highres2standard.mat
    """
    get_standard2highres = pe.MapNode(
            interface = fsl.ConvertXFM(invert_xfm = True),
            iterfield = ['in_file'],
            name = 'get_standard2highres')
    registration.connect(linear_highres2standard,'out_matrix_file',
                         get_standard2highres,'in_file')
    registration.connect(get_standard2highres,'out_file',
                         outputnode,'standard2highres_mat')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
        -omat example_func2standard.mat -concat highres2standard.mat example_func2highres.mat
    """
    get_exmaple_func2standard = pe.MapNode(
            interface = fsl.ConvertXFM(concat_xfm = True),
            iterfield = ['in_file','in_file2'],
            name = 'get_exmaple_func2standard')
    registration.connect(linear_example_func2highres, 'out_matrix_file',
                         get_exmaple_func2standard,'in_file')
    registration.connect(linear_highres2standard,'out_matrix_file',
                         get_exmaple_func2standard,'in_file2')
    registration.connect(get_exmaple_func2standard,'out_file',
                         outputnode,'example_func2standard_mat')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convertwarp 
        --ref=standard 
        --premat=example_func2highres.mat 
        --warp1=highres2standard_warp 
        --out=example_func2standard_warp
    """
    convertwarp_example2standard = pe.MapNode(
            interface = fsl.ConvertWarp(),
            iterfield = ['reference','premat','warp1'],
            name = 'convertwarp_example2standard')
    registration.connect(inputnode,'standard',
                         convertwarp_example2standard,'reference')
    registration.connect(linear_example_func2highres,'out_matrix_file',
                         convertwarp_example2standard,'premat')
    registration.connect(nonlinear_highres2standard,'fieldcoeff_file',
                         convertwarp_example2standard,'warp1')
    registration.connect(convertwarp_example2standard,'out_file',
                         outputnode,'example_func2standard_warp_nii_gz')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/applywarp 
        --ref=standard 
        --in=example_func 
        --out=example_func2standard 
        --warp=example_func2standard_warp
    """
    warp_example2stand = pe.MapNode(
            interface = fsl.ApplyWarp(),
            iterfield = ['ref_file','in_file','field_file'],
            name = 'warp_example2stand')
    registration.connect(inputnode,'standard',
                         warp_example2stand,'ref_file')
    registration.connect(inputnode,'example_func',
                         warp_example2stand,'in_file')
    registration.connect(warp_example2stand,'out_file',
                         outputnode,'example_func2standard_nii_gz')
    registration.connect(convertwarp_example2standard,'out_file',
                         warp_example2stand,'field_file')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
        -inverse -omat standard2example_func.mat example_func2standard.mat
    """
    get_standard2example_func = pe.MapNode(
            interface = fsl.ConvertXFM(invert_xfm = True),
            iterfield = ['in_file'],
            name = 'get_standard2example_func')
    registration.connect(get_exmaple_func2standard,'out_file',
                         get_standard2example_func,'in_file')
    registration.connect(get_standard2example_func,'out_file',
                         outputnode,'standard2example_func_mat')
    
    registration.base_dir = output_dir
    
    registration.inputs.inputspec.highres = anat_brain
    registration.inputs.inputspec.highres_head= anat_head
    registration.inputs.inputspec.example_func = example_func
    registration.inputs.inputspec.standard = standard_brain
    registration.inputs.inputspec.standard_head = standard_head
    registration.inputs.inputspec.standard_mask = standard_mask
    
    # define all the oupput file names with the directory
    registration.inputs.linear_example_func2highres.out_file          = os.path.abspath(os.path.join(output_dir,
                            'example_func2highres.nii.gz'))
    registration.inputs.linear_example_func2highres.out_matrix_file   = os.path.abspath(os.path.join(output_dir,
                            'example_func2highres.mat'))
    registration.inputs.linear_example_func2highres.out_log           = os.path.abspath(os.path.join(output_dir,
                            'linear_example_func2highres.log'))
    registration.inputs.get_highres2example_func.out_file        = os.path.abspath(os.path.join(output_dir,
                            'highres2example_func.mat'))
    registration.inputs.linear_highres2standard.out_file         = os.path.abspath(os.path.join(output_dir,
                            'highres2standard_linear.nii.gz'))
    registration.inputs.linear_highres2standard.out_matrix_file  = os.path.abspath(os.path.join(output_dir,
                            'highres2standard.mat'))
    registration.inputs.linear_highres2standard.out_log          = os.path.abspath(os.path.join(output_dir,
                            'linear_highres2standard.log'))
    # --iout
    registration.inputs.nonlinear_highres2standard.warped_file  = os.path.abspath(os.path.join(output_dir,
                            'highres2standard.nii.gz'))
    # --cout
    registration.inputs.nonlinear_highres2standard.fieldcoeff_file    = os.path.abspath(os.path.join(output_dir,
                            'highres2standard_warp.nii.gz'))
    # --jout
    registration.inputs.nonlinear_highres2standard.jacobian_file      = os.path.abspath(os.path.join(output_dir,
                            'highres2highres_jac.nii.gz'))
    registration.inputs.nonlinear_highres2standard.log_file           = os.path.abspath(os.path.join(output_dir,
                            'nonlinear_highres2standard.log'))
    registration.inputs.warp_highres2standard.out_file                = os.path.abspath(os.path.join(output_dir,
                            'highres2standard.nii.gz'))
    registration.inputs.get_standard2highres.out_file       = os.path.abspath(os.path.join(output_dir,
                            'standard2highres.mat'))
    registration.inputs.get_exmaple_func2standard.out_file               = os.path.abspath(os.path.join(output_dir,
                            'example_func2standard.mat'))
    registration.inputs.convertwarp_example2standard.out_file     = os.path.abspath(os.path.join(output_dir,
                            'example_func2standard_warp.nii.gz'))
    registration.inputs.warp_example2stand.out_file       = os.path.abspath(os.path.join(output_dir,
                            'example_func2standard.nii.gz'))
    registration.inputs.get_standard2example_func.out_file       = os.path.abspath(os.path.join(output_dir,
                            'standard2example_func.mat'))
    return registration

def _create_registration_workflow(anat_brain,
                                 anat_head,
                                 func_ref,
                                 standard_brain,
                                 standard_head,
                                 standard_mask,
                                 output_dir = 'temp'):
    from nipype.interfaces          import fsl
    """
    fslmaths /bcbl/home/public/Consciousness/uncon_feat/data/MRI/sub-01/anat/sub-01-T1W_mprage_sag_p2_1iso_MGH_day_6_nipy_brain highres
    fslmaths /bcbl/home/public/Consciousness/uncon_feat/data/MRI/sub-01/anat/sub-01-T1W_mprage_sag_p2_1iso_MGH_day_6_nipy_brain  highres_head
    fslmaths /opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm_brain standard
    fslmaths /opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm standard_head
    fslmaths /opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm_brain_mask_dil standard_mask
    
    """
    fslmaths = fsl.ImageMaths()
    fslmaths.inputs.in_file = anat_brain
    fslmaths.inputs.out_file = os.path.abspath(os.path.join(output_dir,'highres.nii.gz'))
    fslmaths.cmdline
    fslmaths.run()
    
    fslmaths = fsl.ImageMaths()
    fslmaths.inputs.in_file = anat_head
    fslmaths.inputs.out_file = os.path.abspath(os.path.join(output_dir,'highres_head.nii.gz'))
    fslmaths.cmdline
    fslmaths.run()
    
    fslmaths = fsl.ImageMaths()
    fslmaths.inputs.in_file = standard_brain
    fslmaths.inputs.out_file = os.path.abspath(os.path.join(output_dir,'standard.nii.gz'))
    fslmaths.cmdline
    fslmaths.run()
    
    fslmaths = fsl.ImageMaths()
    fslmaths.inputs.in_file = standard_head
    fslmaths.inputs.out_file = os.path.abspath(os.path.join(output_dir,'standard_head.nii.gz'))
    fslmaths.cmdline
    fslmaths.run()
    
    fslmaths = fsl.ImageMaths()
    fslmaths.inputs.in_file = standard_mask
    fslmaths.inputs.out_file = os.path.abspath(os.path.join(output_dir,'standard_mask.nii.gz'))
    fslmaths.cmdline
    fslmaths.run()
    
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/flirt 
        -in example_func 
        -ref highres 
        -out example_func2highres 
        -omat example_func2highres.mat 
        -cost corratio 
        -dof 7 
        -searchrx -180 180 
        -searchry -180 180 
        -searchrz -180 180 
        -interp trilinear 
    """
    flt = fsl.FLIRT()
    flt.inputs.in_file = func_ref
    flt.inputs.reference = anat_brain
    flt.inputs.out_file = os.path.abspath(os.path.join(output_dir,'example_func2highres.nii.gz'))
    flt.inputs.out_matrix_file = os.path.abspath(os.path.join(output_dir,'example_func2highres.mat'))
    flt.inputs.out_log = os.path.abspath(os.path.join(output_dir,'example_func2highres.log'))
    flt.inputs.cost = 'corratio'
    flt.inputs.interp = 'trilinear'
    flt.inputs.searchr_x = [-180, 180]
    flt.inputs.searchr_y = [-180, 180]
    flt.inputs.searchr_z = [-180, 180]
    flt.inputs.dof = 7
    flt.inputs.save_log = True
    flt.cmdline
    flt.run()
    
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
        -inverse -omat highres2example_func.mat example_func2highres.mat
    """
    inverse_transformer = fsl.ConvertXFM()
    inverse_transformer.inputs.in_file = os.path.abspath(os.path.join(output_dir,"example_func2highres.mat"))
    inverse_transformer.inputs.invert_xfm = True
    inverse_transformer.inputs.out_file = os.path.abspath(os.path.join(output_dir,'highres2example_func.mat'))
    inverse_transformer.cmdline
    inverse_transformer.run()
    
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/flirt 
        -in highres 
        -ref standard 
        -out highres2standard 
        -omat highres2standard.mat 
        -cost corratio 
        -dof 12 
        -searchrx -180 180 
        -searchry -180 180 
        -searchrz -180 180 
        -interp trilinear 
    
    """
    flt = fsl.FLIRT()
    flt.inputs.in_file = anat_brain
    flt.inputs.reference = standard_brain
    flt.inputs.out_file = os.path.abspath(os.path.join(output_dir,'highres2standard_linear.nii.gz'))
    flt.inputs.out_matrix_file = os.path.abspath(os.path.join(output_dir,'highres2standard.mat'))
    flt.inputs.out_log = os.path.abspath(os.path.join(output_dir,'highres2standard.log'))
    flt.inputs.cost = 'corratio'
    flt.inputs.interp = 'trilinear'
    flt.inputs.searchr_x = [-180, 180]
    flt.inputs.searchr_y = [-180, 180]
    flt.inputs.searchr_z = [-180, 180]
    flt.inputs.dof = 12
    flt.inputs.save_log = True
    flt.cmdline
    flt.run()
    
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/fnirt 
        --iout=highres2standard_head 
        --in=highres_head 
        --aff=highres2standard.mat 
        --cout=highres2standard_warp 
        --iout=highres2standard 
        --jout=highres2highres_jac 
        --config=T1_2_MNI152_2mm 
        --ref=standard_head 
        --refmask=standard_mask 
        --warpres=10,10,10
    """
    
    fnirt_mprage = fsl.FNIRT()
    fnirt_mprage.inputs.warp_resolution = (10, 10, 10)
    # --iout name of output image
    fnirt_mprage.inputs.warped_file = os.path.abspath(os.path.join(output_dir,
                                                                 'highres2standard.nii.gz'))
    # --in input image
    fnirt_mprage.inputs.in_file = anat_head
    # --aff affine transform
    fnirt_mprage.inputs.affine_file = os.path.abspath(os.path.join(output_dir,
                                                                   'highres2standard.mat'))
    # --cout output file with field coefficients
    fnirt_mprage.inputs.fieldcoeff_file = os.path.abspath(os.path.join(output_dir,
                                                                       'highres2standard_warp.nii.gz'))
    # --jout
    fnirt_mprage.inputs.jacobian_file = os.path.abspath(os.path.join(output_dir,
                                                                     'highres2highres_jac.nii.gz'))
    # --config
    fnirt_mprage.inputs.config_file = 'T1_2_MNI152_2mm'
    # --ref
    fnirt_mprage.inputs.ref_file = os.path.abspath(standard_head)
    # --refmask
    fnirt_mprage.inputs.refmask_file = os.path.abspath(standard_mask)
    # --warpres
    fnirt_mprage.inputs.log_file = os.path.abspath(os.path.join(output_dir,
                                                                'highres2standard.log'))
    fnirt_mprage.cmdline
    fnirt_mprage.run()
    
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/applywarp 
        -i highres 
        -r standard 
        -o highres2standard 
        -w highres2standard_warp
    """
    aw = fsl.ApplyWarp()
    aw.inputs.in_file = anat_brain
    aw.inputs.ref_file = os.path.abspath(standard_brain)
    aw.inputs.out_file = os.path.abspath(os.path.join(output_dir,
                                                      'highres2standard.nii.gz'))
    aw.inputs.field_file = os.path.abspath(os.path.join(output_dir,
                                                        'highres2standard_warp.nii.gz'))
    aw.cmdline
    aw.run()
    
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
        -inverse -omat standard2highres.mat highres2standard.mat
    """
    inverse_transformer = fsl.ConvertXFM()
    inverse_transformer.inputs.in_file = os.path.abspath(os.path.join(output_dir,"highres2standard.mat"))
    inverse_transformer.inputs.invert_xfm = True
    inverse_transformer.inputs.out_file = os.path.abspath(os.path.join(output_dir,'standard2highres.mat'))
    inverse_transformer.cmdline
    inverse_transformer.run()
    
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
        -omat example_func2standard.mat -concat highres2standard.mat example_func2highres.mat
    """
    inverse_transformer = fsl.ConvertXFM()
    inverse_transformer.inputs.in_file2 = os.path.abspath(os.path.join(output_dir,"highres2standard.mat"))
    inverse_transformer.inputs.in_file = os.path.abspath(os.path.join(output_dir,
                                                                       "example_func2highres.mat"))
    inverse_transformer.inputs.concat_xfm = True
    inverse_transformer.inputs.out_file = os.path.abspath(os.path.join(output_dir,'example_func2standard.mat'))
    inverse_transformer.cmdline
    inverse_transformer.run()
    
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convertwarp 
        --ref=standard 
        --premat=example_func2highres.mat 
        --warp1=highres2standard_warp 
        --out=example_func2standard_warp
    """
    warputils = fsl.ConvertWarp()
    warputils.inputs.reference = os.path.abspath(standard_brain)
    warputils.inputs.premat = os.path.abspath(os.path.join(output_dir,
                                                           "example_func2highres.mat"))
    warputils.inputs.warp1 = os.path.abspath(os.path.join(output_dir,
                                                          "highres2standard_warp.nii.gz"))
    warputils.inputs.out_file = os.path.abspath(os.path.join(output_dir,
                                                             "example_func2standard_warp.nii.gz"))
    warputils.cmdline
    warputils.run()
    
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/applywarp 
        --ref=standard 
        --in=example_func 
        --out=example_func2standard 
        --warp=example_func2standard_warp
    """
    aw = fsl.ApplyWarp()
    aw.inputs.ref_file = os.path.abspath(standard_brain)
    aw.inputs.in_file = os.path.abspath(func_ref)
    aw.inputs.out_file = os.path.abspath(os.path.join(output_dir,
                                                      "example_func2standard.nii.gz"))
    aw.inputs.field_file = os.path.abspath(os.path.join(output_dir,
                                                        "example_func2standard_warp.nii.gz"))
    aw.run()
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
        -inverse -omat standard2example_func.mat example_func2standard.mat
    """
    inverse_transformer = fsl.ConvertXFM()
    inverse_transformer.inputs.in_file = os.path.abspath(os.path.join(output_dir,
                                                               "example_func2standard.mat"))
    inverse_transformer.inputs.out_file = os.path.abspath(os.path.join(output_dir,
                                                                "standard2example_func.mat"))
    inverse_transformer.inputs.invert_xfm = True
    inverse_transformer.cmdline
    inverse_transformer.run()
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
    /opt/fsl/fsl-5.0.10/fsl/bin/slicer {example_func2highres} {highres} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
    /opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2highres}1.png ; 
    /opt/fsl/fsl-5.0.10/fsl/bin/slicer {highres} {example_func2highres} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
    /opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2highres}2.png ; 
    /opt/fsl/fsl-5.0.10/fsl/bin/pngappend {example_func2highres}1.png - {example_func2highres}2.png {example_func2highres}.png; 
    /bin/rm -f sl?.png {example_func2highres}2.png
    /bin/rm {example_func2highres}1.png
    """.replace("\n"," ")
    
    plot_highres2standard = f"""
    /opt/fsl/fsl-5.0.10/fsl/bin/slicer {highres2standard} {standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
    /opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {highres2standard}1.png ; 
    /opt/fsl/fsl-5.0.10/fsl/bin/slicer {standard} {highres2standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
    /opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {highres2standard}2.png ; 
    /opt/fsl/fsl-5.0.10/fsl/bin/pngappend {highres2standard}1.png - {highres2standard}2.png {highres2standard}.png; 
    /bin/rm -f sl?.png {highres2standard}2.png
    /bin/rm {highres2standard}1.png
    """.replace("\n"," ")
    
    plot_example_func2standard = f"""
    /opt/fsl/fsl-5.0.10/fsl/bin/slicer {example_func2standard} {standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
    /opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2standard}1.png ; 
    /opt/fsl/fsl-5.0.10/fsl/bin/slicer {standard} {example_func2standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
    /opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2standard}2.png ; 
    /opt/fsl/fsl-5.0.10/fsl/bin/pngappend {example_func2standard}1.png - {example_func2standard}2.png {example_func2standard}.png; 
    /bin/rm -f sl?.png {example_func2standard}2.png
    """.replace("\n"," ")
    for cmdline in [plot_example_func2highres,plot_example_func2standard,plot_highres2standard]:
        os.system(cmdline)


def create_simple_struc2BOLD(roi,
                             roi_name,
                             preprocessed_functional_dir,
                             output_dir):
    from nipype.interfaces            import fsl
    from nipype.pipeline              import engine as pe
    from nipype.interfaces            import utility as util
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
    
    simple_workflow         = pe.Workflow(name  = 'struc2BOLD')
    
    inputnode               = pe.Node(interface = util.IdentityInterface(
                                      fields    = ['flt_in_file',
                                                   'flt_in_matrix',
                                                   'flt_reference',
                                                   'mask']),
                                      name      = 'inputspec')
    outputnode              = pe.Node(interface = util.IdentityInterface(
                                      fields    = ['BODL_mask']),
                                      name      = 'outputspec')
    """
     flirt 
 -in /export/home/dsoto/dsoto/fmri/$s/sess2/label/$i 
 -ref /export/home/dsoto/dsoto/fmri/$s/sess2/run1_prepro1.feat/example_func.nii.gz  
 -applyxfm 
 -init /export/home/dsoto/dsoto/fmri/$s/sess2/run1_prepro1.feat/reg/highres2example_func.mat 
 -out  /export/home/dsoto/dsoto/fmri/$s/label/BOLD${i}
    """
    flirt_convert           = pe.MapNode(
                                    interface   = fsl.FLIRT(apply_xfm = True),
                                    iterfield   = ['in_file',
                                                   'reference',
                                                   'in_matrix_file'],
                                    name        = 'flirt_convert')
    simple_workflow.connect(inputnode,      'flt_in_file',
                            flirt_convert,  'in_file')
    simple_workflow.connect(inputnode,      'flt_reference',
                            flirt_convert,  'reference')
    simple_workflow.connect(inputnode,      'flt_in_matrix',
                            flirt_convert,  'in_matrix_file')
    
    """
     fslmaths /export/home/dsoto/dsoto/fmri/$s/label/BOLD${i} -mul 2 
     -thr `fslstats /export/home/dsoto/dsoto/fmri/$s/label/BOLD${i} -p 99.6` 
    -bin /export/home/dsoto/dsoto/fmri/$s/label/BOLD${i}
    """
    def getthreshop(thresh):
        return ['-mul 2 -thr %.10f -bin' % (val) for val in thresh]
    getthreshold            = pe.MapNode(
                                    interface   = fsl.ImageStats(op_string='-p 99.6'),
                                    iterfield   = ['in_file','mask_file'],
                                    name        = 'getthreshold')
    simple_workflow.connect(flirt_convert,  'out_file',
                            getthreshold,   'in_file')
    simple_workflow.connect(inputnode,      'mask',
                            getthreshold,   'mask_file')
    
    threshold               = pe.MapNode(
                                    interface   = fsl.ImageMaths(
                                            suffix      = '_thresh',
                                            op_string   = '-mul 2 -bin'),
                                    iterfield   = ['in_file','op_string'],
                                    name        = 'thresholding')
    simple_workflow.connect(flirt_convert,  'out_file',
                            threshold,      'in_file')
    simple_workflow.connect(getthreshold,   ('out_stat',getthreshop),
                            threshold,      'op_string')
#    simple_workflow.connect(threshold,'out_file',outputnode,'BOLD_mask')
    
    bound_by_mask           = pe.MapNode(
                                    interface   = fsl.ImageMaths(
                                            suffix      = '_mask',
                                            op_string   = '-mas'),
                                    iterfield   = ['in_file','in_file2'],
                                    name        = 'bound_by_mask')
    simple_workflow.connect(threshold,      'out_file',
                            bound_by_mask,  'in_file')
    simple_workflow.connect(inputnode,      'mask',
                            bound_by_mask,  'in_file2')
    simple_workflow.connect(bound_by_mask,  'out_file',
                            outputnode,     'BOLD_mask')
    
    # setup inputspecs 
    simple_workflow.inputs.inputspec.flt_in_file    = roi
    simple_workflow.inputs.inputspec.flt_in_matrix  = os.path.abspath(os.path.join(preprocessed_functional_dir,
                                                        'reg',
                                                        'highres2example_func.mat'))
    simple_workflow.inputs.inputspec.flt_reference  = os.path.abspath(os.path.join(preprocessed_functional_dir,
                                                        'func',
                                                        'example_func.nii.gz'))
    simple_workflow.inputs.inputspec.mask           = os.path.abspath(os.path.join(preprocessed_functional_dir,
                                                        'func',
                                                        'mask.nii.gz'))
    simple_workflow.inputs.bound_by_mask.out_file   = os.path.abspath(os.path.join(output_dir,
                                                         roi_name.replace('_fsl.nii.gz',
                                                                          '_BOLD.nii.gz')))
    return simple_workflow

def registration_plotting(output_dir,
                          anat_brain,
                          standard_brain):
    ######################
    ###### plotting ######
    try:
        example_func2highres    = os.path.abspath(os.path.join(output_dir,
                                                'example_func2highres'))
        example_func2standard   = os.path.abspath(os.path.join(output_dir,
                                                 'example_func2standard_warp'))
        highres2standard        = os.path.abspath(os.path.join(output_dir,
                                                 'highres2standard'))
        highres                 = os.path.abspath(anat_brain)
        standard                = os.path.abspath(standard_brain)
        
        plot_example_func2highres = f"""
/opt/fsl/fsl-5.0.10/fsl/bin/slicer {example_func2highres} {highres} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2highres}1.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/slicer {highres} {example_func2highres} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2highres}2.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/pngappend {example_func2highres}1.png - {example_func2highres}2.png {example_func2highres}.png; 
/bin/rm -f sl?.png {example_func2highres}2.png
/bin/rm {example_func2highres}1.png
        """.replace("\n"," ")
        
        plot_highres2standard = f"""
/opt/fsl/fsl-5.0.10/fsl/bin/slicer {highres2standard} {standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {highres2standard}1.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/slicer {standard} {highres2standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {highres2standard}2.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/pngappend {highres2standard}1.png - {highres2standard}2.png {highres2standard}.png; 
/bin/rm -f sl?.png {highres2standard}2.png
/bin/rm {highres2standard}1.png
        """.replace("\n"," ")
        
        plot_example_func2standard = f"""
/opt/fsl/fsl-5.0.10/fsl/bin/slicer {example_func2standard} {standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2standard}1.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/slicer {standard} {example_func2standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2standard}2.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/pngappend {example_func2standard}1.png - {example_func2standard}2.png {example_func2standard}.png; 
/bin/rm -f sl?.png {example_func2standard}2.png
        """.replace("\n"," ")
        for cmdline in [plot_example_func2highres,
                        plot_example_func2standard,
                        plot_highres2standard]:
            os.system(cmdline)
    except:
        print('you should not use python 2.7, update your python!!')

def create_highpass_filter_workflow(workflow_name = 'highpassfiler',
                                    HP_freq = 60,
                                    TR = 0.85):
    from nipype.workflows.fmri.fsl    import preprocess
    from nipype.interfaces            import fsl
    from nipype.pipeline              import engine as pe
    from nipype.interfaces            import utility as util
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
    getthreshop         = preprocess.getthreshop
    getmeanscale        = preprocess.getmeanscale
    highpass_workflow = pe.Workflow(name = workflow_name)
    
    inputnode               = pe.Node(interface = util.IdentityInterface(
                                      fields    = ['ICAed_file',]),
                                      name      = 'inputspec')
    outputnode              = pe.Node(interface = util.IdentityInterface(
                                      fields    = ['filtered_file']),
                                      name      = 'outputspec')
    
    img2float = pe.MapNode(interface    = fsl.ImageMaths(out_data_type     = 'float',
                                                         op_string         = '',
                                                         suffix            = '_dtype'),
                           iterfield    = ['in_file'],
                           name         = 'img2float')
    highpass_workflow.connect(inputnode,'ICAed_file',
                              img2float,'in_file')
    
    getthreshold = pe.MapNode(interface     = fsl.ImageStats(op_string = '-p 2 -p 98'),
                              iterfield     = ['in_file'],
                              name          = 'getthreshold')
    highpass_workflow.connect(img2float,    'out_file',
                              getthreshold, 'in_file')
    thresholding = pe.MapNode(interface     = fsl.ImageMaths(out_data_type  = 'char',
                                                             suffix         = '_thresh',
                                                             op_string      = '-Tmin -bin'),
                                iterfield   = ['in_file','op_string'],
                                name        = 'thresholding')
    highpass_workflow.connect(img2float,    'out_file',
                              thresholding, 'in_file')
    highpass_workflow.connect(getthreshold,('out_stat',getthreshop),
                              thresholding,'op_string')
    
    dilatemask = pe.MapNode(interface   = fsl.ImageMaths(suffix     = '_dil',
                                                         op_string  = '-dilF'),
                            iterfield   = ['in_file'],
                            name        = 'dilatemask')
    highpass_workflow.connect(thresholding,'out_file',
                              dilatemask,'in_file')
    
    maskfunc = pe.MapNode(interface     = fsl.ImageMaths(suffix     = '_mask',
                                                         op_string  = '-mas'),
                          iterfield     = ['in_file','in_file2'],
                          name          = 'apply_dilatemask')
    highpass_workflow.connect(img2float,    'out_file',
                              maskfunc,     'in_file')
    highpass_workflow.connect(dilatemask,   'out_file',
                              maskfunc,     'in_file2')
    
    medianval = pe.MapNode(interface    = fsl.ImageStats(op_string = '-k %s -p 50'),
                           iterfield    = ['in_file','mask_file'],
                           name         = 'cal_intensity_scale_factor')
    highpass_workflow.connect(img2float,    'out_file',
                              medianval,    'in_file')
    highpass_workflow.connect(thresholding, 'out_file',
                              medianval,    'mask_file')
    
    meanscale = pe.MapNode(interface    = fsl.ImageMaths(suffix = '_intnorm'),
                           iterfield    = ['in_file','op_string'],
                           name         = 'meanscale')
    highpass_workflow.connect(maskfunc,     'out_file',
                              meanscale,    'in_file')
    highpass_workflow.connect(medianval,    ('out_stat',getmeanscale),
                              meanscale,    'op_string')
    
    meanfunc = pe.MapNode(interface     = fsl.ImageMaths(suffix     = '_mean',
                                                         op_string  = '-Tmean'),
                           iterfield    = ['in_file'],
                           name         = 'meanfunc')
    highpass_workflow.connect(meanscale, 'out_file',
                              meanfunc,  'in_file')
    
    
    hpf = pe.MapNode(interface  = fsl.ImageMaths(suffix     = '_tempfilt',
                                                 op_string  = '-bptf %.10f -1' % (HP_freq/2/TR)),
                     iterfield  = ['in_file'],
                     name       = 'highpass_filering')
    highpass_workflow.connect(meanscale,'out_file',
                              hpf,      'in_file',)
    
    addMean = pe.MapNode(interface  = fsl.BinaryMaths(operation = 'add'),
                         iterfield  = ['in_file','operand_file'],
                         name       = 'addmean')
    highpass_workflow.connect(hpf,      'out_file',
                              addMean,  'in_file')
    highpass_workflow.connect(meanfunc, 'out_file',
                              addMean,  'operand_file')
    
    highpass_workflow.connect(addMean,      'out_file',
                              outputnode,   'filtered_file')
    
    return highpass_workflow























