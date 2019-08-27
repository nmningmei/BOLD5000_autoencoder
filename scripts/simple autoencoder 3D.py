#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 12:43:41 2019

@author: nmei

simple autoencoder

"""

import os
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset
from nibabel import load as load_fmri

from torch import nn,no_grad
from torch.nn import functional
import torch.optim as optim
from torch.autograd import Variable

class customizedDataset(Dataset):
    def __init__(self,data_root):
        self.samples = []
        
        for item in glob(os.path.join(data_root,'*.nii.gz')):
            self.samples.append(item)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return load_fmri(self.samples[idx]).get_data() / load_fmri(self.samples[idx]).get_data().max()

class encoder3D(nn.Module):
    def __init__(self,
                 batch_size         = 10,
                 in_channels        = [1,16,32],
                 out_channels       = [16,32,32],
                 kernel_size        = 3,
                 stride             = 1,
                 padding_mode       = 'valid',
                 pool_kernal_size   = 2,
                 ):
        super(encoder3D, self).__init__()
        
        self.batch_size             = batch_size
        self.in_channels            = in_channels
        self.out_channels           = out_channels
        self.kernel_size            = kernel_size
        self.stride                 = stride
        self.padding_mode           = padding_mode
        self.pool_kernal_size       = pool_kernal_size
        
        self.conv3d_1_16 = nn.Conv3d(in_channels    = self.in_channels[0],
                                     out_channels   = self.out_channels[0],
                                     kernel_size    = self.kernel_size,
                                     stride         = self.stride,
                                     padding_mode   = self.padding_mode,
                                     )
        self.conv3d_16_32 = nn.Conv3d(in_channels   = self.in_channels[1],
                                      out_channels  = self.out_channels[1],
                                      kernel_size   = self.kernel_size,
                                      stride        = self.stride,
                                      padding_mode  = self.padding_mode,
                                      )
        self.conv3d_32_32 = nn.Conv3d(in_channels   = self.in_channels[2],
                                      out_channels  = self.out_channels[2],
                                      kernel_size   = self.kernel_size,
                                      stride        = self.stride,
                                      padding_mode  = self.padding_mode,
                                      )
        self.activation = nn.LeakyReLU(inplace      = True)
        self.pooling    = nn.AvgPool3d(kernel_size     = self.pool_kernal_size,
                                    stride          = 2,
                                    )
        self.norm16 = nn.BatchNorm3d(num_features   = 16)
        self.norm32 = nn.BatchNorm3d(num_features   = 32)
        
    def forward(self,x):
        out1 = self.norm16(self.conv3d_1_16(x))
        out1 = self.activation(out1)
        out1 = self.norm32(self.conv3d_16_32(out1))
        out1 = self.activation(out1)
        out1 = self.pooling(out1)
        out2 = self.norm32(self.conv3d_32_32(out1))
        out2 = self.activation(out2)
        out2 = self.norm32(self.conv3d_32_32(out2))
        out2 = self.activation(out2)
        out2 = self.pooling(out2)
        out3 = self.norm32(self.conv3d_32_32(out2))
        out3 = self.activation(out3)
        out3 = self.norm32(self.conv3d_32_32(out3))
        out3 = self.activation(out3)
        out3 = self.pooling(out3)
#        print(out3.shape)
#        out4 = self.activation(self.conv3d_32_32(out3))
#        out4 = self.norm32(out4)
#        out4 = self.pooling(out4)
#        print(out4.shape)
        
        flatten = out3.view(self.batch_size,-1)
        
        return flatten
class decoder3D(nn.Module):
    def __init__(self,
                 batch_size     = 10,
                 in_channels    = [32,16,1],
                 out_channels   = [32,16,1],
                 kernel_size    = 10,
                 stride         = 1,
                 padding_mode   = 'zeros',):
        super(decoder3D, self).__init__()
        
        self.batch_size         = batch_size
        self.in_channels        = in_channels
        self.out_channels       = out_channels
        self.kernel_size        = kernel_size
        self.stride             = stride
        self.padding_mode       = padding_mode
        
        self.convT3d_32_16 = nn.ConvTranspose3d(in_channels     = self.in_channels[0],
                                                out_channels    = self.out_channels[1],
                                                kernel_size     = self.kernel_size,
                                                stride          = self.stride,
                                                padding_mode    = self.padding_mode)
        self.convT3d_16_16 = nn.ConvTranspose3d(in_channels     = self.in_channels[1],
                                                out_channels    = self.out_channels[1],
                                                kernel_size     = self.kernel_size,
                                                stride          = self.stride,
                                                padding_mode    = self.padding_mode)
        self.convT3d_16_1 = nn.ConvTranspose3d(in_channels      = self.in_channels[1],
                                               out_channels     = self.out_channels[2],
                                               kernel_size      = self.kernel_size,
                                               stride           = self.stride,
                                               padding_mode     = self.padding_mode)
        self.convT3d_1_1 = nn.ConvTranspose3d(in_channels       = self.in_channels[2],
                                              out_channels      = self.out_channels[2],
                                              kernel_size       = self.kernel_size,#(2,2,2),
                                              stride            = self.stride,#(1,1,1),
                                              padding_mode      = self.padding_mode)
        
        
        self.activation         = nn.LeakyReLU(inplace          = True)
        self.output_activation  = nn.Softsign()
        self.norm               = nn.BatchNorm3d(num_features   = 1)
        self.norm16             = nn.BatchNorm3d(num_features   = 16)
        
    def forward(self,x):
        reshaped = x.view(self.batch_size,32,7,7,4)
        
        out1 = self.norm16(self.convT3d_32_16(reshaped))
        out1 = self.activation(out1)
        out1 = functional.interpolate(out1,size = (24,24,18))
        
        out2 = self.norm16(self.convT3d_16_16(out1))
        out2 = self.activation(out2)
        out2 = functional.interpolate(out2,size = (40,40,30))
        
        out3 = self.norm16(self.convT3d_16_16(out2))
        out3 = self.activation(out3)
        out3 = functional.interpolate(out3,size = (56,56,42))
        
        out4 = self.norm(self.convT3d_16_1(out3))
        out4 = self.activation(out4)
        out4 = functional.interpolate(out4,size = (88,88,66))
        out4 = self.output_activation(out4)
        
        return out4

def createLossAndOptimizer(net, learning_rate=0.001):
    
    #Loss function
    loss        = nn.MSELoss()
    
    #Optimizer
    optimizer   = optim.Adam(net.parameters(), lr = learning_rate)
    
    return(loss, optimizer)

def train_loop(net,loss_fuc,optimizer,dataloader,idx_epoch = 1):
    """
    A for-loop of train the autoencoder for 1 epoch
    """
    train_loss      = 0.
    for ii,batch in enumerate(dataloader):
        try:
            # load the data to memory
            if torch.cuda.is_available():
                inputs  = Variable(batch.unsqueeze(1)).cuda()
            else:
                inputs  = Variable(batch.unsqueeze(1))
            # one of the most important step, reset the gradients
            optimizer.zero_grad()
            # compute the outputs
            outputs     = net(inputs)
            # compute the losses
            loss_batch  = loss_func(outputs.squeeze(1),inputs.squeeze(1),)
            # backpropagation
            loss_batch.backward()
            # modify the weights
            optimizer.step()
            # record the training loss of a mini-batch
            train_loss  += loss_batch.data
            print(f'epoch {idx_epoch+1}-{ii + 1:3.0f}/{100*(ii+1)/len(dataloader):.3f}%,loss = {train_loss/(ii+1):.6f}')
        except Exception as e:
            print(e)
    return train_loss

def validation_loop(net,loss_func,dataloader,idx_epoch = 1):
    # specify the gradient being frozen
    with no_grad():
        valid_loss      = 0.
        for ii,batch in tqdm(enumerate(dataloader)):
            try:
                # load the data to memory
                if torch.cuda.is_available():
                    inputs  = Variable(batch.unsqueeze(1)).cuda()
                else:
                    inputs  = Variable(batch.unsqueeze(1))
                # compute the outputs
                outputs     = autoencoder(inputs)
                # compute the losses
                loss_batch  = loss_func(outputs.squeeze(1),inputs.squeeze(1),)
                # record the validation loss of a mini-batch
                valid_loss  += loss_batch.data
                denominator = ii
            except Exception as e:
                print(e)
        valid_loss = valid_loss / (denominator + 1)
    return valid_loss

if __name__ == '__main__':
    print('import')
    from torch.utils.data import DataLoader
    import torch
    import numpy as np
    import pandas as pd
    print()
    saving_name     = '../results/simple_autoencoder3D.pth'
    
    batch_size      = 10
    lr              = 1e-3
    n_epochs        = 200
    print('set up random seeds')
    torch.manual_seed(12345)
    if torch.cuda.is_available():torch.cuda.empty_cache();torch.cuda.manual_seed(12345)
    print('set up data loaders')
    train_dataset   = customizedDataset('../data/train/')
    valid_dataset   = customizedDataset('../data/validation/')
    dataloader_train = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=2)
    dataloader_valid = DataLoader(valid_dataset, batch_size = batch_size, shuffle=False,num_workers=2)
    print('set up autoencoder')
    encoder             = encoder3D()
    decoder             = decoder3D()#;rec = decoder(out);print(rec.shape)
    if torch.cuda.is_available():
        autoencoder     = nn.Sequential(*[encoder,decoder]).cuda()
    else:
        autoencoder     = nn.Sequential(*[encoder,decoder])
    loss_func,optimizer = createLossAndOptimizer(autoencoder,learning_rate = lr)
    
    print('initialize')
    results = dict(
            train_loss  = [],
            valid_loss  = [],
            epochs      = [])
    
    best_valid_loss         = torch.from_numpy(np.array(np.inf))
    for idx_epoch in range(n_epochs):
        # train
        train_loss          = train_loop(autoencoder,loss_func,optimizer,dataloader_train,idx_epoch)
        # validation
        if idx_epoch > 0:
            encoder         = encoder3D()
            decoder         = decoder3D()
            if torch.cuda.is_available():
                autoencoder = nn.Sequential(*[encoder,decoder]).cuda()
            else:
                autoencoder = nn.Sequential(*[encoder,decoder])
            autoencoder.load_state_dict(torch.load(saving_name))
            autoencoder.eval()
        valid_loss = validation_loop(autoencoder,loss_func,dataloader_valid,idx_epoch)
    
        print(f'epoch {idx_epoch + 1}, validation loss = {valid_loss:.6f}')
        
        if torch.tensor(valid_loss.cpu(),dtype=torch.float64) < best_valid_loss:
            best_valid_loss = torch.tensor(valid_loss.cpu().clone().detach(),dtype=torch.float64)
            torch.save(autoencoder.state_dict(),saving_name)
            print('saving model')
        results['train_loss'].append(train_loss)
        results['valid_loss'].append(valid_loss)
        results['epochs'].append(idx_epoch + 1)
        results_to_save = pd.DataFrame(results)
        results_to_save.to_csv(saving_name.replace(".pth",".csv"),index = False)



































