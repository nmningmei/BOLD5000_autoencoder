# BOLD5000_autoencoder
autoencoder trained on BOLD5000 dataset

# [Data: BOLD500](https://bold5000.github.io)

link to download the data: https://figshare.com/articles/BOLD5000/6459449

Please note: only "Unfiltered" data were used.

# preprocessing pipeline
## motion correction, susan smoothing etc
![prefmri](https://github.com/nmningmei/BOLD5000_autoencoder/blob/master/figures/preprocessing_step_1.png)
## ICA AROMA denoising
```
# get the subject name
sub_name = np.unique(re.findall(r'CSI\d',picked_data))[0]
# get the session and run
n_session = np.unique(re.findall(r'Sess-\d+',picked_data))[0]
n_run = np.unique(re.findall(r'Run-\d+',picked_data))[0]
# get the file of the first run of the first session
run1 = 'Run-1_'
session1 = 'Sess-1_'
first_run = os.path.abspath([item for item in glob(os.path.join(data_dir,
                              "*",
                              "*",
                              "*.nii.gz")) if \
            (sub_name in item)\
            and (session1 in item)\
            and (run1 in item)][0])
first_run_dir = '/'.join(first_run.split('/')[:-1])
# define some of the kind inputs
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
# put the inputs to corresponding argument placeholders
AROMA_obj           = ICA_AROMA()
AROMA_obj.inputs.in_file            = os.path.abspath(preprocessed_fmri)
AROMA_obj.inputs.mat_file           = os.path.abspath(func_to_struct)
AROMA_obj.inputs.fnirt_warp_file    = os.path.abspath(warpfield)
AROMA_obj.inputs.motion_parameters  = os.path.abspath(fsl_mcflirt_movpar)
AROMA_obj.inputs.mask               = os.path.abspath(mask)
AROMA_obj.inputs.denoise_type       = 'nonaggr'
AROMA_obj.inputs.out_dir            = os.path.abspath(output_dir)
# with "-ow" is to overwrite the old results
cmdline             = 'python ../ICA_AROMA/' + AROMA_obj.cmdline + ' -ow'

os.system(cmdline)
```
## register functional scans to structural scans
![reg](https://github.com/nmningmei/BOLD5000_autoencoder/blob/master/figures/registrate%20funtional%20scans%20to%20sctural%20scans.png)
##
![hpf](https://github.com/nmningmei/BOLD5000_autoencoder/blob/master/figures/highpass_temp.png)
