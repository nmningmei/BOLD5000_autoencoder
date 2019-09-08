#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 10:54:14 2019

@author: nmei
"""

import os
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset
from nibabel import load as load_fmri

import torch
from torch import nn
from torch.autograd import Variable
from torchvision import models

class customizedDataset(Dataset):
    def __init__(self,data_root):
        self.samples = []
        for item in glob(os.path.join(data_root,'*.nii.gz')):
            self.samples.append(item)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):# for each item in the sample
        """
        Because we will use binary cross entropy loss function, 
        it is crutial that we make sure the data range between 0 and 1
        """
        # load the nii.gz format data
        temp = load_fmri(self.samples[idx]).get_data()
        # get the maximum of the volume
        max_weight = temp.max()
        # ge the minmum of the volume
        min_weight = temp.min()
        # standardize
        temp_std = (temp - min_weight) / (max_weight - min_weight)
        temp_scaled = temp_std * (1 - 0) + 0
        return temp_scaled,max_weight,min_weight,self.samples[idx]


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    from torch.nn.modules.module import _addindent

    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr 
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'   

    tmpstr = tmpstr + ')'
    return tmpstr
    
class encoder2D(nn.Module):
    def __init__(self,
                 batch_size         = 10,
                 device             = 'cpu'):
        super(encoder2D, self).__init__()
        
        self.batch_size = batch_size
        self.adaptive_pooling = nn.AdaptiveAvgPool2d((1,1))
        self.device = device
    
    def forward(self,x):
        import numpy as np
        iterator = np.arange(66).reshape(-1,3)
        
        preallocator = torch.zeros((self.batch_size,iterator.shape[0],1280))
        for ii,slices in enumerate(iterator):
            x_sliced = x[:,slices]
            pretrained = models.mobilenet_v2(pretrained = True).to(device).features
            for params in pretrained:
                params.requires_grad = False
            temp_out = pretrained(x_sliced)
            temp_out = torch.squeeze(self.adaptive_pooling(temp_out))
            preallocator[:,ii] = temp_out
        return preallocator

if __name__ == '__main__':
    print('import')
    from torch.utils.data import DataLoader
    import numpy as np
    print()
    saving_name     = '../results/decoder2D.pth'
    
    batch_size      = 1
    print('set up random seeds')
    torch.manual_seed(12345)
    if torch.cuda.is_available():torch.cuda.empty_cache();torch.cuda.manual_seed(12345);
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device:{device}')
    print('set up data loaders')
    train_dataset   = customizedDataset('../data/train/')
    valid_dataset   = customizedDataset('../data/validation/')
    dataloader_train = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=2)
    dataloader_valid = DataLoader(valid_dataset, batch_size = batch_size, shuffle=False,num_workers=2)
    print('set up autoencoder')
    encoder = encoder2D(batch_size = batch_size,
                        device = device)#;out = encoder(next(iter(dataloader_train))[0].permute(0,3,1,2));print(out.shape);#asdf
    for ii,(batch,maximum,minimum,name) in tqdm(enumerate(dataloader_train)):
        name = name[0]
        name = name.split('/')[-1]
        name = name.replace('.nii.gz','.npy')
        saving_name = '../data/rep_train/{}'.format(name)
        if not os.path.exists(saving_name):
            batch = Variable(batch).to(device)
            rep = encoder(batch)
            np.save(saving_name,rep.detach().cpu().numpy()[0],)
    for ii,(batch,maximum,minimum,name) in tqdm(enumerate(dataloader_valid)):
        name = name[0]
        name = name.split('/')[-1]
        name = name.replace('.nii.gz','.npy')
        saving_name = '../data/rep_validation/{}'.format(name)
        if not os.path.exists(saving_name):
            batch = Variable(batch).to(device)
            rep = encoder(batch)
            np.save(saving_name,rep.detach().cpu().numpy()[0],)
