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
            self.samples.append(item)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return load_fmri(self.samples[idx]).get_data() / load_fmri(self.samples[idx]).get_data().max()

class encoder2D(nn.Module):
    def __init__(self,
                 batch_size         = 10,
                 in_channels        = [66, 128,256,512],
                 out_channels       = [128,256,512],
                 kernel_size        = 3,
                 stride             = 1,
                 padding_mode       = 'valid',
                 pool_kernal_size   = 2,
                 ):
        super(encoder2D, self).__init__()
        
        self.batch_size             = batch_size
        self.in_channels            = in_channels
        self.out_channels           = out_channels
        self.kernel_size            = kernel_size
        self.stride                 = stride
        self.padding_mode           = padding_mode
        self.pool_kernal_size       = pool_kernal_size
        
        self.conv2d_66_128 = nn.Conv2d(in_channels    = self.in_channels[0],
                                       out_channels   = self.out_channels[0],
                                       kernel_size    = self.kernel_size,
                                       stride         = self.stride,
                                       padding_mode   = self.padding_mode,
                                       )
        self.conv2d_128_128 = nn.Conv2d(in_channels    = self.in_channels[1],
                                        out_channels   = self.out_channels[0],
                                        kernel_size    = self.kernel_size,
                                        stride         = self.stride,
                                        padding_mode   = self.padding_mode,
                                        )
        self.conv2d_128_256 = nn.Conv2d(in_channels    = self.in_channels[1],
                                        out_channels   = self.out_channels[1],
                                        kernel_size    = self.kernel_size,
                                        stride         = self.stride,
                                        padding_mode   = self.padding_mode,
                                        )
        self.conv2d_256_256 = nn.Conv2d(in_channels    = self.in_channels[2],
                                        out_channels   = self.out_channels[1],
                                        kernel_size    = self.kernel_size,
                                        stride         = self.stride,
                                        padding_mode   = self.padding_mode,
                                        )
        self.conv2d_256_512 = nn.Conv2d(in_channels    = self.in_channels[2],
                                        out_channels   = self.out_channels[2],
                                        kernel_size    = self.kernel_size,
                                        stride         = self.stride,
                                        padding_mode   = self.padding_mode,
                                        )
        self.conv2d_512_512 = nn.Conv2d(in_channels    = self.in_channels[3],
                                        out_channels   = self.out_channels[2],
                                        kernel_size    = self.kernel_size,
                                        stride         = self.stride,
                                        padding_mode   = self.padding_mode,
                                        )
        
        self.activation = nn.LeakyReLU(inplace      = True)
        self.pooling = nn.AvgPool2d(kernel_size     = self.pool_kernal_size,
                                    stride          = 2,
                                    )
        self.output_pooling = nn.AdaptiveAvgPool2d(output_size = (1,1))
        self.norm128 = nn.BatchNorm2d(num_features   = 128)
        self.norm256 = nn.BatchNorm2d(num_features   = 256)
        self.norm512 = nn.BatchNorm2d(num_features   = 512)
        
    def forward(self,x):
        out1 = self.norm128(self.conv2d_66_128(x))
        out1 = self.activation(out1)
        out1 = self.norm128(self.conv2d_128_128(out1))
        out1 = self.activation(out1)
        out1 = self.pooling(out1)
        
        out2 = self.norm256(self.conv2d_128_256(out1))
        out2 = self.activation(out2)
        out2 = self.norm256(self.conv2d_256_256(out2))
        out2 = self.activation(out2)
        out2 = self.pooling(out2)
        
        out3 = self.norm512(self.conv2d_256_512(out2))
        out3 = self.activation(out3)
        out3 = self.norm512(self.conv2d_512_512(out3))
        out3 = self.activation(out3)
        out3 = self.pooling(out3)
        
        flatten = torch.squeeze(self.output_pooling(out3))
        
        return flatten
class decoder2D(nn.Module):
    def __init__(self,
                 batch_size     = 10,
                 in_channels    = [512,256,128,66],
                 out_channels   = [512,256,128,66],
                 kernel_size    = 7,
                 stride         = 1,
                 padding_mode   = 'zeros',):
        super(decoder2D, self).__init__()
        
        self.batch_size         = batch_size
        self.in_channels        = in_channels
        self.out_channels       = out_channels
        self.kernel_size        = kernel_size
        self.stride             = stride
        self.padding_mode       = padding_mode
        
        self.convT2d_512_512    = nn.ConvTranspose2d(in_channels    = self.in_channels[0],
                                                     out_channels   = self.out_channels[0],
                                                     kernel_size    = self.kernel_size,
                                                     stride         = self.stride,
                                                     padding_mode   = self.padding_mode,)
        self.convT2d_512_256    = nn.ConvTranspose2d(in_channels    = self.in_channels[0],
                                                     out_channels   = self.out_channels[1],
                                                     kernel_size    = self.kernel_size,
                                                     stride         = self.stride,
                                                     padding_mode   = self.padding_mode,)
        
        self.convT2d_256_256    = nn.ConvTranspose2d(in_channels    = self.in_channels[1],
                                                     out_channels   = self.out_channels[1],
                                                     kernel_size    = self.kernel_size,
                                                     stride         = self.stride,
                                                     padding_mode   = self.padding_mode,)
        self.convT2d_256_128    = nn.ConvTranspose2d(in_channels    = self.in_channels[1],
                                                     out_channels   = self.out_channels[2],
                                                     kernel_size    = self.kernel_size,
                                                     stride         = self.stride,
                                                     padding_mode   = self.padding_mode,)
        
        self.convT2d_128_128    = nn.ConvTranspose2d(in_channels    = self.in_channels[2],
                                                     out_channels   = self.out_channels[2],
                                                     kernel_size    = self.kernel_size,
                                                     stride         = self.stride,
                                                     padding_mode   = self.padding_mode,)
        self.convT2d_128_66    = nn.ConvTranspose2d(in_channels     = self.in_channels[2],
                                                     out_channels   = self.out_channels[3],
                                                     kernel_size    = self.kernel_size,
                                                     stride         = self.stride,
                                                     padding_mode   = self.padding_mode,)
        
        self.convT2d_66_66    = nn.ConvTranspose2d(in_channels      = self.in_channels[3],
                                                     out_channels =  self.out_channels[3],
                                                     kernel_size    = self.kernel_size,
                                                     stride         = self.stride,
                                                     padding_mode   = self.padding_mode,)
        
        self.activation         = nn.LeakyReLU(inplace              = True)
        self.output_activation  = nn.Softsign()
        self.norm512                = nn.BatchNorm2d(num_features   = 512)
        self.norm256                = nn.BatchNorm2d(num_features   = 256)
        self.norm128                = nn.BatchNorm2d(num_features   = 128)
        self.norm66                 = nn.BatchNorm2d(num_features   = 66)
        
    def forward(self,x):
        reshaped = x.view(self.batch_size,512,1,1)
        
        out1 = self.norm512(self.convT2d_512_512(reshaped))
        out1 = self.norm512(self.convT2d_512_512(out1))
        out1 = self.norm256(self.convT2d_512_256(out1))
        out1 = self.activation(out1)
        out1 = functional.interpolate(out1,size = (24,24))
        
        out2 = self.norm256(self.convT2d_256_256(out1))
        out2 = self.norm256(self.convT2d_256_256(out2))
        out2 = self.norm128(self.convT2d_256_128(out2))
        out2 = self.activation(out2)
        out2 = functional.interpolate(out2,size = (48,48))
        
        out3 = self.norm128(self.convT2d_128_128(out2))
        out3 = self.norm128(self.convT2d_128_128(out3))
        out3 = self.norm66(self.convT2d_128_66(out3))
        out3 = self.activation(out3)
        # no need to interpolate because it is 66 x 66 x 66
        
        out4 = self.norm66(self.convT2d_66_66(out3))
        out4 = self.norm66(self.convT2d_66_66(out4))
        out4 = self.norm66(self.convT2d_66_66(out4))
        out4 = self.activation(out4)
        out4 = functional.interpolate(out4,size = (88,88))
        
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
        print(f'epoch {idx_epoch}-{ii + 1},{train_loss/(ii+1):.4f}')
        
    return train_loss
def validation_loop(net,loss_func,dataloader,idx_epoch = 1):
    # specify the gradient being frozen
    with no_grad():
        valid_loss      = 0.
        for ii,batch in enumerate(dataloader):
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
        valid_loss = valid_loss / (ii + 1)
    return valid_loss
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import torch
    import numpy as np
    
    train_dataset   = customizedDataset('../data/train/')
    valid_dataset   = customizedDataset('../data/validation/')
    
    saving_name     = '../results/simple_autoencoder2D.pth'
    
    batch_size      = 10
    lr              = 1e-3
    n_epochs        = 200
    
    torch.cuda.empty_cache()
    torch.manual_seed(12345)
    if torch.cuda.is_available():torch.cuda.manual_seed(12345)
    
    dataloader_train = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=2)
    dataloader_valid = DataLoader(valid_dataset, batch_size = batch_size, shuffle=False,num_workers=2)
    encoder = encoder2D()
    decoder = decoder2D()
    if torch.cuda.is_available():
        autoencoder     = nn.Sequential(*[encoder,decoder]).cuda()
    else:
        autoencoder     = nn.Sequential(*[encoder,decoder])
    loss_func,optimizer = createLossAndOptimizer(autoencoder,learning_rate = lr)
    
    best_valid_loss         = torch.from_numpy(np.array(np.inf))
    for idx_epoch in range(n_epochs):
        # train
        train_loss          = train_loop(autoencoder,loss_func,optimizer,dataloader_train,idx_epoch)
        # validation
        if idx_epoch > 0:
            encoder         = encoder2D()
            decoder         = decoder2D()
            if torch.cuda.is_available():
                autoencoder = nn.Sequential(*[encoder,decoder]).cuda()
            else:
                autoencoder = nn.Sequential(*[encoder,decoder])
            autoencoder.load_state_dict(torch.load(saving_name))
            autoencoder.eval()
        valid_loss = validation_loop(autoencoder,loss_func,dataloader_valid,idx_epoch)
    
    print(f'epoch {idx_epoch}, validation loss = {valid_loss:.4f}')
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss.copy()
        torch.save(autoencoder.state_dict(),saving_name)
        print('saving model')



































