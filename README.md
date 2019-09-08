# BOLD5000_autoencoder
An autoencoder trained on the BOLD5000 dataset

# Inspiration
Even when participants view images of living and non-living objects that they are not able to discriminate behaviorally, we are able to classify the correct categories based on the functional magnetic resonance imaging blood-oxygen-level-dependent (fMRI BOLD) responses above chance level. In short, the BOLD response provides more information than behavior in classifying the categories. While significant, the area under the receiver operating characteristic curve (ROC AUC) is low, so we sought to utilize transfer learning \cite {yosinski2014transferable} to improve the performance. Yosinski et al.’s work was based on supervised machine learning, but a system trained with unsupervised machine learning is more generalizable \cite {bengio2012deep}. Thus we trained an autoencoder which is an unsupervised machine learning algorithm that consists of two parts: an encoder and a decoder. An encoder extracts “features” that compresses the input, and then the decoder reconstructs the input based on the extracted “features”.

# Requirements
- Python 3.5+
- mricrogl - convert dcm to nii.gz, dcm2niix package
- mricon - convert dcm to nii.gz, particularly for structural scans, dcm2nii package
- FSL 5.0.10
- Freesurfer 6.0.0 (optional for structural scan processing)
- AFNI - stable version

# Python Libraries
- numpy 1.16.1
- pandas 0.20.3
- nipype 1.1.9
- nibabel 2.4.1
- nilearn 0.5.0
- scikit-learning 0.20.3
- pytorch-cpu 1.2

# Checklist:
- [x] Download and preprocess the BOLD5000 dataset
- [x] Get relevant volumes from the preprocessed data
- [x] Train a simple autoencoder with the data
- [ ] Train a variational autoencoder with the data
- [ ] Train a convolutional neural network using the pretrained simple autoencoder with the BOLD data and images
- [ ] Examine the generalizability of the simple/variational autoencoder to a new fMRI dataset, using the autoencoder as a "feature extractor"

# Step 0. Download data
Link to download the data: https://figshare.com/articles/BOLD5000/6459449

Please note: only "Unfiltered" data were used.

## Step 0.1. Unzip

## Step 0.2. Convert dcm to nii.gz (nipype, mricrogl and mricon are needed)
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

# Step 1. Preprocessing pipeline of the functional scans (nipype, fsl, and freesurfer are needed)

## Step 1.1. [MCFLIRT, susan smoothing etc. For details of the pipeline, click here](https://colab.research.google.com/github/nmningmei/preprocessing_pipelines/blob/master/FSL_vs_nipype_fsl_preprocessing.ipynb#scrollTo=QF77EkMJrrrI)
![prefmri](https://github.com/nmningmei/BOLD5000_autoencoder/blob/master/figures/preprocessing_step_1.png)

## Step 1.2. [ICA AROMA](https://www.ncbi.nlm.nih.gov/pubmed/25770991) and [denoising](https://github.com/maartenmennes/ICA-AROMA) - you have to download the GitHub repository to go with the data (nipype, fsl, and ICA AROMA github repo needed)
```
from nipype.interfaces.fsl import ICA_AROMA
# get the subject name
sub_name = np.unique(re.findall(r'CSI\d',picked_data))[0]
# get the session and run
n_session = np.unique(re.findall(r'Sess-\d+',picked_data))[0]
n_run = np.unique(re.findall(r'Run-\d+',picked_data))[0]
# get the file of the first run of the first session. Why? Because ICA AROMA takes the structural files and structural scans are only processed when the first run of the first session is processed. 
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
# inputs to the corresponding argument placeholders
AROMA_obj           = ICA_AROMA()
AROMA_obj.inputs.in_file            = os.path.abspath(preprocessed_fmri)
AROMA_obj.inputs.mat_file           = os.path.abspath(func_to_struct)
AROMA_obj.inputs.fnirt_warp_file    = os.path.abspath(warpfield)
AROMA_obj.inputs.motion_parameters  = os.path.abspath(fsl_mcflirt_movpar)
AROMA_obj.inputs.mask               = os.path.abspath(mask)
AROMA_obj.inputs.denoise_type       = 'nonaggr'
AROMA_obj.inputs.out_dir            = os.path.abspath(output_dir)
# with "-ow" you can overwrite the old results
cmdline             = 'python ../ICA_AROMA/' + AROMA_obj.cmdline + ' -ow'
# run the graph we set up above
os.system(cmdline)
```

## Step 1.3. Register functional scans to structural scans (nipype, fsl, and freesurfer needed)
![reg](https://github.com/nmningmei/BOLD5000_autoencoder/blob/master/figures/registrate%20funtional%20scans%20to%20sctural%20scans.png)

## Step 1.4. High pass filter at 60 Hz (nipype and fsl are needed)
![hpf](https://github.com/nmningmei/BOLD5000_autoencoder/blob/master/figures/highpass_temp.png)

## Step 1.5. Reshape the volumes into 88 x 88 x 66 with larger voxel sizes
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

## Step 1.6. Get relevant volumes
![fmri-protocol](https://github.com/nmningmei/BOLD5000_autoencoder/blob/master/figures/fMRI%20protocol.jpg)
According to the figure shown above, we don't have to use all the MRI volumes but a small subset of them because not all of them are related to the images shown to the subjects. Here we pick those volumes 4 - 8 seconds after the onset of the image at each trial. 

# Step 2. Autoencoder modeling

## Step 2.1. Simple autoencoder - compression and reconstruction
![simple-autoencoder](https://github.com/nmningmei/BOLD5000_autoencoder/blob/master/figures/autoencoder%20phase%201.jpg)

- batch size: 16
- initial learning rate: 1e-2, decreased by 5 after every 4 epochs
- maximum epochs: 200
- training size: 44320
- validation size: 11080
- best validation loss: 0.008989973 (MSE)

## Step 2.2. Variational autoencoder - variational inference and reconstruction
<img align="center" width="100" height="100" src="https://github.com/nmningmei/BOLD5000_autoencoder/blob/master/figures/vae_model.png">
[source: pyro documentation](http://pyro.ai/examples/vae.html#Variational-Autoencoders)

![vae](https://github.com/nmningmei/BOLD5000_autoencoder/blob/master/figures/autoencoder%20phase%202.jpg)

## Step 2.3. Generalization to a different experiment - cross experiment generalization (test data is available upon request)
![generalize](https://github.com/nmningmei/BOLD5000_autoencoder/blob/master/figures/autoencoder%20phase%203.jpg)

## Step 2.4. Train a convolutional neural network which outputs the same latent representation - which is an approximation of an image-to-BOLD mapper (what the brain does)
![mapper](https://github.com/nmningmei/BOLD5000_autoencoder/blob/master/figures/autoencoder%20phase%204.jpg)
