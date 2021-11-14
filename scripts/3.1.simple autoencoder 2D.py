#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 12:27:23 2021

@author: nmei
"""
import os

import numpy as np

from torch.utils.data import DataLoader
import torch
from torch import nn,optim
from torch.autograd import Variable
from torchvision import models
from torch.utils.data import Dataset

from utils_deep import (customizedDataset,
                        Simple2DEncoder,
                        train_loop,
                        valid_loop
                        )

if __name__ == '__main__':
    saving_name             = '../models/SAE2D.pth'
    batch_size              = 2
    pretrained_model_name   = 'mobilenet'
    learning_rate           = 1e-2
    n_epochs                = 10
    print('set up random seeds')
    torch.manual_seed(12345)
    if torch.cuda.is_available():torch.cuda.empty_cache();torch.cuda.manual_seed(12345);
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device:{device}')
    print('set up data loaders')
    train_dataset    = customizedDataset('../data/train/')
    valid_dataset    = customizedDataset('../data/validation/')
    dataloader_train = DataLoader(train_dataset, 
                                  batch_size = batch_size, 
                                  shuffle = True, 
                                  num_workers = 1, 
                                  drop_last = True,
                                  )
    dataloader_valid = DataLoader(valid_dataset, 
                                  batch_size = batch_size, 
                                  shuffle = False,
                                  num_workers = 1, 
                                  drop_last = True,
                                  )
    
    print('set up the model')
    encoder = Simple2DEncoder(
        pretrained_model_name,
        batch_size = batch_size,
        device = device,
        pretrained = False,
        )
    model_parameters    = [p for p in encoder.parameters() if p.requires_grad]
    n_params            = sum([np.prod(p.size()) for p in model_parameters])
    print(pretrained_model_name,f'total params = {n_params}')
    
    optimizer = optim.SGD(model_parameters, lr = learning_rate,)
    for idx_epoch in range(n_epochs):
        print('training...')
        _ = train_loop(encoder, dataloader_train, optimizer, device,idx_epoch = idx_epoch,)
        print('validating...')
        valid_loss = valid_loop(encoder,dataloader_valid,device,idx_epoch = idx_epoch,)
        
    
    
