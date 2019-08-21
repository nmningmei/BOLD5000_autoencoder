# BOLD5000_autoencoder
autoencoder trained on BOLD5000 dataset

# Requirements
- Python 3.5+
- mricrogl - convert dcm to nii.gz, dcm2niix package
- mricon - convert dcm to nii.gz, particularly for structural scans, dcm2nii package
- FSL 5.0.10
- Freesurfer 6.0.0 (optional for structural scan processing)
- AFNI - stable version
## Python Libraries
- numpy 1.16.1
- pandas 0.20.3
- nipype 1.1.9
- nibabel 2.4.1
- nilearn 0.5.0
- scikit-learning 0.20.3
- pytorch-cpu 1.2

# Step 0 [Download Data: BOLD500](https://bold5000.github.io)

link to download the data: https://figshare.com/articles/BOLD5000/6459449

Please note: only "Unfiltered" data were used.
## step 0.1.unzip
## step 0.2.convet dcm to nii.gz
```
from nipype.interfaces.dcm2nii import Dcm2niix
converter = Dcm2niix()
converter.inputs.source_dir = os.path.abspath(dcm_files_in_a_folder)
converter.inputs.output_dir = new_output_directory # must create before running converter
converter.inputs.bids_format = True
converter.inputs.single_file = True
converter.inputs.crop = False
print(converter.cmdline)
converter.run()
```

# Step 1.Preprocessing Pipeline - functional scans
## step 1.1.[motion correction, susan smoothing etc. Details of the pipeline, click here.](https://nbviewer.jupyter.org/github/nmningmei/preprocessing_pipelines/blob/master/FSL_vs_nipype_fsl_preprocessing.ipynb)
![prefmri](https://github.com/nmningmei/BOLD5000_autoencoder/blob/master/figures/preprocessing_step_1.png)
## step 1.2.ICA AROMA denoising
```
# get the subject name
sub_name = np.unique(re.findall(r'CSI\d',picked_data))[0]
# get the session and run
n_session = np.unique(re.findall(r'Sess-\d+',picked_data))[0]
n_run = np.unique(re.findall(r'Run-\d+',picked_data))[0]
# get the file of the first run of the first session. Why? Because ICA AROMA takes strutural preprocessed files and structural scans are only processed when the first run of the first session is processed. 
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
## step 1.3.register functional scans to structural scans
![reg](https://github.com/nmningmei/BOLD5000_autoencoder/blob/master/figures/registrate%20funtional%20scans%20to%20sctural%20scans.png)
## step 1.4.high pass filter at 60 Hz
![hpf](https://github.com/nmningmei/BOLD5000_autoencoder/blob/master/figures/highpass_temp.png)

## step 1.4.reshape the volumes into 88 x 88 x 66 with larger voxel size.
```
from nipype.interfaces import afni
from nilearn.image import resample_img
from nibabel import load as load_fmri

target_func = load_fmri(
                os.path.abspath(
                    os.path.join(
                      data_dir,
                      'target_func.nii.gz')
                      )
                      )
# afni 3dresample -dxyz 2.386364,2.386364,2.4 -prefix output.nii.gz -input input.nii.gz
# resample the voxel sizes
resample3d = afni.utils.Resample(voxel_size = (2.386364,2.386364,2.4))
resample3d.inputs.in_file = picked_data
resample3d.inputs.outputtype = 'NIFTI_GZ'
resample3d.inputs.out_file = picked_data.replace('filtered.nii.gz',
                                                 'filtered_resample.nii.gz')
print(resample3d.cmdline)
resample3d.run()

# reshape into 88 by 88 by 66
resampled = resample_img(resample3d.inputs.out_file,
                         target_affine = target_func.affine,
                         target_shape = (88,88,66))
resampled.to_filename(picked_data.replace('filtered.nii.gz',
                                          'filtered_reshaped.nii.gz'))
```
## step 1.5.get relevant volumes
![fmri-protocol](https://github.com/nmningmei/BOLD5000_autoencoder/blob/master/figures/fMRI%20protocol.jpg)

# Step 2 Autoencoder Modeling
## step 2.1.Simple Autoencoder - compression, reconstruction
![simple-autoencoder](https://github.com/nmningmei/BOLD5000_autoencoder/blob/master/figures/autoencoder%20phase%201.jpg)
## step 2.2.Variational Autoencoder - variantional inference, reconstruction

<img align="center" width="100" height="100" src="https://github.com/nmningmei/BOLD5000_autoencoder/blob/master/figures/vae_model.png">

![vae](https://github.com/nmningmei/BOLD5000_autoencoder/blob/master/figures/autoencoder%20phase%202.jpg)
## step 2.3.generalization to a different experiment - cross experiment generalization (test data is available upon requiest)
![generalize](https://github.com/nmningmei/BOLD5000_autoencoder/blob/master/figures/autoencoder%20phase%203.jpg)
## step 2.4.train a convolutional neural network, outputs the same latent representation - approximation of an image-to-BOLD mapper (aka, the brain)
![mapper](https://github.com/nmningmei/BOLD5000_autoencoder/blob/master/figures/autoencoder%20phase%204.jpg)
