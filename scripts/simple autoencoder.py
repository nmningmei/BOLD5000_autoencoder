#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 12:43:41 2019

@author: nmei

simple autoencoder

"""

import os
from glob import glob
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
#            fmri = load_fmri(item)
            self.samples.append(item)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return load_fmri(self.samples[idx]).get_data() / load_fmri(self.samples[idx]).get_data().max()

class encoder3D(nn.Module):
    def __init__(self,
                 batch_size = 10,
                 in_channels = [1,16,32],
                 out_channels = [16,32,32],
                 kernel_size = 3,
                 stride = 1,
                 padding_mode = 'valid',
                 pool_kernal_size = 2,
                 ):
        super(encoder3D, self).__init__()
        
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_mode = padding_mode
        self.pool_kernal_size = pool_kernal_size
        
        self.conv3d_1_16 = nn.Conv3d(in_channels = self.in_channels[0],
                                       out_channels = self.out_channels[0],
                                       kernel_size = self.kernel_size,
                                       stride = self.stride,
                                       padding_mode = self.padding_mode,
                                       )
        self.conv3d_16_32 = nn.Conv3d(in_channels = self.in_channels[1],
                                       out_channels = self.out_channels[1],
                                       kernel_size = self.kernel_size,
                                       stride = self.stride,
                                       padding_mode = self.padding_mode,
                                       )
        self.conv3d_32_32 = nn.Conv3d(in_channels = self.in_channels[2],
                                       out_channels = self.out_channels[2],
                                       kernel_size = self.kernel_size,
                                       stride = self.stride,
                                       padding_mode = self.padding_mode,
                                       )
        self.activation = nn.LeakyReLU(inplace = True)
        self.pooling = nn.AvgPool3d(kernel_size = self.pool_kernal_size,
                                    stride = 2,
                                    )
        self.norm16 = nn.BatchNorm3d(num_features = 16)
        self.norm32 = nn.BatchNorm3d(num_features = 32)
        
    def forward(self,x):
        out1 = self.activation(self.conv3d_1_16(x))
        out1 = self.norm16(out1)
        out1 = self.activation(self.conv3d_16_32(out1))
        out1 = self.norm32(out1)
        out1 = self.pooling(out1)
        out2 = self.activation(self.conv3d_32_32(out1))
        out2 = self.norm32(out2)
        out2 = self.activation(self.conv3d_32_32(out2))
        out2 = self.norm32(out2)
        out2 = self.pooling(out2)
        out3 = self.activation(self.conv3d_32_32(out2))
        out3 = self.norm32(out3)
        out3 = self.activation(self.conv3d_32_32(out3))
        out3 = self.norm32(out3)
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
                 batch_size = 10,
                 in_channels = [32,16,1],
                 out_channels = [32,16,1],
                 kernel_size = 10,
                 stride = 1,
                 padding_mode = 'zeros',):
        super(decoder3D, self).__init__()
        
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_mode = padding_mode
        
        self.convT3d_32_16 = nn.ConvTranspose3d(in_channels = self.in_channels[0],
                                                out_channels = self.out_channels[1],
                                                kernel_size = self.kernel_size,
                                                stride = self.stride,
                                                padding_mode = self.padding_mode)
        self.convT3d_16_16 = nn.ConvTranspose3d(in_channels = self.in_channels[1],
                                                out_channels = self.out_channels[1],
                                                kernel_size = self.kernel_size,
                                                stride = self.stride,
                                                padding_mode = self.padding_mode)
        self.convT3d_16_1 = nn.ConvTranspose3d(in_channels = self.in_channels[1],
                                               out_channels = self.out_channels[2],
                                               kernel_size = self.kernel_size,
                                               stride = self.stride,
                                               padding_mode = self.padding_mode)
        self.convT3d_1_1 = nn.ConvTranspose3d(in_channels = self.in_channels[2],
                                              out_channels = self.out_channels[2],
                                              kernel_size = self.kernel_size,#(2,2,2),
                                              stride = self.stride,#(1,1,1),
                                              padding_mode = self.padding_mode)
        
        
        self.activation = nn.LeakyReLU(inplace = True)
        self.output_activation = nn.Softsign()
        self.norm = nn.BatchNorm3d(num_features = 1)
        self.norm16 = nn.BatchNorm3d(num_features = 16)
        
    def forward(self,x):
        reshaped = x.view(self.batch_size,32,7,7,4)
        
        out1 = self.activation(self.convT3d_32_16(reshaped))
        out1 = self.norm16(out1)
#        print(out1.shape)
        out1 = functional.interpolate(out1,size = (24,24,18))
        
        out2 = self.activation(self.convT3d_16_16(out1))
        out2 = self.norm16(out2)
#        print(out2.shape)
        out2 = functional.interpolate(out2,size = (40,40,30))
        
        out3 = self.activation(self.convT3d_16_16(out2))
        out3 = self.norm16(out3)
#        print(out3.shape)
        out3 = functional.interpolate(out3,size = (56,56,42))
        
        out4 = self.activation(self.convT3d_16_1(out3))
        out4 = self.norm(out4)
#        print(out4.shape)
        out4 = functional.interpolate(out4,size = (88,88,66))
        out4 = self.output_activation(out4)
        
        return out4


def createLossAndOptimizer(net, learning_rate=0.001):
    
    #Loss function
    loss = nn.MSELoss()
    
    #Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    return(loss, optimizer)
def train_loop(net,loss_fuc,optimizer,dataloader,idx_epoch = 1):
    
    train_loss = 0.
    for ii,batch in enumerate(dataloader):
        inputs = Variable(batch.unsqueeze(1)).cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        
        loss_batch = loss_func(outputs.squeeze(1),inputs.squeeze(1),)
        loss_batch.backward()
        optimizer.step()
        train_loss += loss_batch.data
        print(f'epoch {idx_epoch}-{ii + 1},{train_loss/(ii+1):.4f}')
        
    return train_loss
def validation_loop(net,loss_func,dataloader,idx_epoch = 1):
    with no_grad():
        valid_loss = 0.
        for ii,batch in enumerate(dataloader):
            inputs = Variable(batch.unsqueeze(1)).cuda()
            outputs = autoencoder(inputs)
            loss_batch = loss_func(outputs.squeeze(1),inputs.squeeze(1),)
            valid_loss += loss_batch.data
        valid_loss = valid_loss / (ii + 1)
    return valid_loss
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import torch
    
    train_dataset = customizedDataset('../data/train/')
    valid_dataset = customizedDataset('../data/validation/')
    
    batch_size = 10
    lr = 1e-3
    n_epochs = 200
    
    torch.cuda.empty_cache()
    torch.manual_seed(12345)
    if torch.cuda.is_available():torch.cuda.manual_seed(12345)
    
    dataloader_train = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=2)
    dataloader_valid = DataLoader(valid_dataset, batch_size = batch_size, shuffle=False,num_workers=2)
    
    encoder = encoder3D()
    decoder = decoder3D()#;rec = decoder(out);print(rec.shape)
    
    autoencoder = nn.Sequential(*[encoder,decoder]).cuda()
    loss_func,optimizer = createLossAndOptimizer(autoencoder,learning_rate = lr)
    
    for idx_epoch in range(n_epochs):
        
        train_loss = train_loop(autoencoder,loss_func,optimizer,dataloader_train,idx_epoch)
        valid_loss = validation_loop(autoencoder,loss_func,dataloader_valid,idx_epoch)
    
    print(f'epoch {idx_epoch}, validation loss = {valid_loss:.4f}')



































