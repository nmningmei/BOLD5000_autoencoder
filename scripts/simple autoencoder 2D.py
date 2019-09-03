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
from scipy.stats import scoreatpercentile

import torch
from torch import nn,no_grad
from torch.nn import functional
from torch import optim
from torch.autograd import Variable

class customizedDataset(Dataset):
    def __init__(self,data_root):
        self.samples = []
        
        for item in glob(os.path.join(data_root,'*.nii.gz')):
            self.samples.append(item)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        temp = load_fmri(self.samples[idx]).get_data()
#        temp[temp < scoreatpercentile(temp.flatten(),2)] = 0
        max_weight = temp.max()
        temp = temp / max_weight
        min_weight = np.abs(temp.min())
        temp = temp + min_weight
        return temp,max_weight,min_weight

class encoder2D(nn.Module):
    def __init__(self,
                 batch_size         = 10,
                 in_channels        = [66, 128,256,512,1028,2056,2000],
                 out_channels       = [128,256,512,1028,2056,2000],
                 kernel_size        = 3,
                 stride             = 2,
                 padding_mode       = 'valid',
                 pool_kernal_size   = 1,
                 ):
        super(encoder2D, self).__init__()
        
        self.batch_size             = batch_size
        self.in_channels            = in_channels
        self.out_channels           = out_channels
        self.kernel_size            = kernel_size
        self.stride                 = stride
        self.padding_mode           = padding_mode
        self.pool_kernal_size       = pool_kernal_size
        
        self.conv2d_0_1 = nn.Conv2d(in_channels    = self.in_channels[0],
                                    out_channels   = self.out_channels[0],
                                    kernel_size    = self.kernel_size,
                                    stride         = self.stride,
                                    padding_mode   = self.padding_mode,
                                    )
        self.conv2d_1_1 = nn.Conv2d(in_channels    = self.in_channels[1],
                                    out_channels   = self.out_channels[0],
                                    kernel_size    = self.kernel_size,
                                    stride         = self.stride,
                                    padding_mode   = self.padding_mode,
                                    )
        self.conv2d_1_2 = nn.Conv2d(in_channels    = self.in_channels[1],
                                    out_channels   = self.out_channels[1],
                                    kernel_size    = self.kernel_size,
                                    stride         = self.stride,
                                    padding_mode   = self.padding_mode,
                                    )
        self.conv2d_2_2 = nn.Conv2d(in_channels    = self.in_channels[2],
                                    out_channels   = self.out_channels[1],
                                    kernel_size    = self.kernel_size,
                                    stride         = self.stride,
                                    padding_mode   = self.padding_mode,
                                    )
        self.conv2d_2_3 = nn.Conv2d(in_channels    = self.in_channels[2],
                                    out_channels   = self.out_channels[2],
                                    kernel_size    = self.kernel_size,
                                    stride         = self.stride,
                                    padding_mode   = self.padding_mode,
                                    )
        self.conv2d_3_3 = nn.Conv2d(in_channels    = self.in_channels[3],
                                    out_channels   = self.out_channels[2],
                                    kernel_size    = self.kernel_size,
                                    stride         = self.stride,
                                    padding_mode   = self.padding_mode,
                                    )
        self.conv2d_3_4 = nn.Conv2d(in_channels    = self.in_channels[3],
                                    out_channels   = self.out_channels[3],
                                    kernel_size    = self.kernel_size,
                                    stride         = self.stride,
                                    padding_mode   = self.padding_mode,
                                    )
        self.conv2d_4_4 = nn.Conv2d(in_channels    = self.in_channels[4],
                                    out_channels   = self.out_channels[3],
                                    kernel_size    = self.kernel_size,
                                    stride         = self.stride,
                                    padding_mode   = self.padding_mode,
                                    )
        self.conv2d_4_5 = nn.Conv2d(in_channels    = self.in_channels[4],
                                    out_channels   = self.out_channels[4],
                                    kernel_size    = self.kernel_size,
                                    stride         = self.stride,
                                    padding_mode   = self.padding_mode,
                                    )
        self.conv2d_5_5 = nn.Conv2d(in_channels    = self.in_channels[5],
                                    out_channels   = self.out_channels[4],
                                    kernel_size    = self.kernel_size,
                                    stride         = self.stride,
                                    padding_mode   = self.padding_mode,
                                    )
        
        self.activation = nn.CELU(inplace      = True)
        self.output_activation = nn.ELU()
        self.pooling = nn.AvgPool2d(kernel_size         = self.pool_kernal_size,
                                    stride              = 1,
                                    count_include_pad   = False
                                    )
        self.output_pooling = nn.AdaptiveAvgPool2d(output_size = (1,1))
        self.norm1 = nn.BatchNorm2d(num_features    = out_channels[0])
        self.norm2 = nn.BatchNorm2d(num_features    = out_channels[1])
        self.norm3 = nn.BatchNorm2d(num_features    = out_channels[2])
        self.norm4 = nn.BatchNorm2d(num_features    = out_channels[3])
        self.norm5 = nn.BatchNorm2d(num_features    = out_channels[4])
        self.dropout = nn.Dropout2d(p = 0.1)
        
    def forward(self,x):
        out1 = self.norm1(self.conv2d_0_1(x))
        out1 = self.activation(out1)
        out1 = self.dropout(out1)
        out2 = self.norm1(self.conv2d_1_1(out1))
        out2 = self.activation(out2)
        out2 = self.pooling(out2)
        
        out3 = self.norm2(self.conv2d_1_2(out2))
        out3 = self.activation(out3)
        out3 = self.dropout(out3)
        out4 = self.norm2(self.conv2d_2_2(out3))
        out4 = self.activation(out4)
        out4 = self.pooling(out4)
        
        out5 = self.norm3(self.conv2d_2_3(out4))
        out5 = self.output_activation(out5)
        
        flatten = torch.squeeze(out5)
        
        return flatten
class decoder2D(nn.Module):
    def __init__(self,
                 batch_size     = 10,
                 in_channels    = [512,256,128,128,66],
                 out_channels   = [512,256,128,128,66],
                 kernel_size    = 7,
                 stride         = 2,
                 padding_mode   = 'zeros',):
        super(decoder2D, self).__init__()
        
        self.batch_size         = batch_size
        self.in_channels        = in_channels
        self.out_channels       = out_channels
        self.kernel_size        = kernel_size
        self.stride             = stride
        self.padding_mode       = padding_mode
        
        self.convT2d_0_1    = nn.ConvTranspose2d(in_channels    = self.in_channels[0],
                                                 out_channels   = self.out_channels[1],
                                                 kernel_size    = self.kernel_size,
                                                 stride         = self.stride,
                                                 padding_mode   = self.padding_mode,)
        
        self.convT2d_1_2    = nn.ConvTranspose2d(in_channels    = self.in_channels[1],
                                                 out_channels   = self.out_channels[2],
                                                 kernel_size    = self.kernel_size,
                                                 stride         = self.stride,
                                                 padding_mode   = self.padding_mode,)
        
        self.convT2d_2_3    = nn.ConvTranspose2d(in_channels     = self.in_channels[2],
                                                 out_channels   = self.out_channels[3],
                                                 kernel_size    = self.kernel_size,
                                                 stride         = self.stride,
                                                 padding_mode   = self.padding_mode,)
        self.convT2d_3_4    = nn.ConvTranspose2d(in_channels     = self.in_channels[3],
                                                 out_channels   = self.out_channels[4],
                                                 kernel_size    = self.kernel_size,
                                                 stride         = self.stride,
                                                 padding_mode   = self.padding_mode,)
        self.convT2d_4_5    = nn.Conv2d(in_channels    = self.in_channels[4],
                                        out_channels   = self.out_channels[4],
                                        kernel_size    = 4,
                                        stride         = 1, 
                                        padding_mode   = 'zeros')
        
        self.activation         = nn.CELU(inplace              = True)
        self.output_activation  = nn.Sigmoid()
        self.norm0                = nn.BatchNorm2d(num_features   = out_channels[0])
        self.norm1                = nn.BatchNorm2d(num_features   = out_channels[1])
        self.norm2                = nn.BatchNorm2d(num_features   = out_channels[2])
        self.norm3                = nn.BatchNorm2d(num_features   = out_channels[3])
        self.norm4                = nn.BatchNorm2d(num_features   = out_channels[4])
        self.dropout = nn.Dropout2d(p = 0.1)
        
    def forward(self,x):
        reshaped = x.view(self.batch_size,self.out_channels[0],1,1)
        
        out1 = self.norm1(self.convT2d_0_1(reshaped))
        out1 = self.activation(out1)
        out1 = self.dropout(out1)
        
        out2 = self.norm2(self.convT2d_1_2(out1))
        out2 = self.activation(out2)
        out2 = self.dropout(out2)
        
        out3 = self.norm3(self.convT2d_2_3(out2))
        out3 = self.activation(out3)
        out3 = self.dropout(out3)
        
        out4 = self.norm4(self.convT2d_3_4(out3))
        out4 = self.activation(out4)
        
        out5 = self.norm4(self.convT2d_4_5(out4))
        out5 = self.output_activation(out5)
        
        
        return out5


def createLossAndOptimizer(net, learning_rate=0.001):
    
    #Loss function
    loss        = nn.BCELoss()
    
    #Optimizer
    optimizer   = optim.Adam(net.parameters(), lr = learning_rate)#,weight_decay = 1e-7)
    
    return(loss, optimizer)

def train_loop(net,loss_fuc,optimizer,dataloader,device,stp,idx_epoch = 1,epsilon = 1e-12):
    """
    A for-loop of train the autoencoder for 1 epoch
    """
    train_loss      = 0.
    for ii,(batch,_,_) in enumerate(dataloader):
        if ii + 1 < len(dataloader):
            # load the data to memory
            inputs  = Variable(batch).to(device)
            # one of the most important step, reset the gradients
            optimizer.zero_grad()
            # compute the outputs
            outputs     = net(inputs.permute(0,3,1,2))
            # compute the losses
            loss_batch  = loss_func(outputs,inputs.permute(0,3,1,2),) # prediction loss
            loss_batch += 0.001 * torch.norm(outputs,1) + epsilon # L1 prediction penalty
            selected_params = torch.cat([x.view(-1) for x in net.parameters()]) # L2 penalty on parameters
            loss_batch += 0.001 * (0.5 * torch.norm(selected_params,1) + 0.5 * torch.norm(selected_params,2) + epsilon)
            # backpropagation
            loss_batch.backward()
            # modify the weights
            optimizer.step()
            # record the training loss of a mini-batch
            train_loss  += loss_batch.data
            print(f'epoch {idx_epoch+stp}-{ii + 1:3.0f}/{100*(ii+1)/ len(dataloader):2.3f}%,loss = {train_loss/(ii+1):.6f}')
    return train_loss/(ii+1)

def validation_loop(net,loss_func,dataloader,device,idx_epoch = 1):
    # specify the gradient being frozen
    with no_grad():
        valid_loss      = 0.
        for ii,(batch,_,_) in tqdm(enumerate(dataloader)):
            if ii + 1 < len(dataloader):
                # load the data to memory
                inputs  = Variable(batch).to(device)
                # compute the outputs
                outputs     = autoencoder(inputs.permute(0,3,1,2))
                # compute the losses
                loss_batch  = loss_func(outputs,inputs.permute(0,3,1,2),)
                # record the validation loss of a mini-batch
                valid_loss  += loss_batch.data
                denominator = ii
        valid_loss = valid_loss / (denominator + 1)
    return valid_loss

if __name__ == '__main__':
    print('import')
    from torch.utils.data import DataLoader
    from collections import OrderedDict
#    import torch
    import numpy as np
    import pandas as pd
    print()
    saving_name     = '../results/simple_autoencoder2D.pth'
    
    batch_size      = 16
    lr              = 1e-4 
    n_epochs        = 10
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
    encoder = encoder2D(batch_size = batch_size)#;out = encoder(next(iter(dataloader_train))[0].permute(0,3,1,2));print(out.shape)#;asdf
    decoder = decoder2D(batch_size = batch_size)#;c = decoder(out);print(c.shape);adf
    
    autoencoder     = nn.Sequential(OrderedDict(
            [('encoder',encoder),
             ('decoder',decoder),
                    ]
            )).to(device)
    if os.path.exists(saving_name.replace(".pth",".csv")):
        autoencoder.load_state_dict(torch.load(saving_name))
        autoencoder.eval()
        results = pd.read_csv(saving_name.replace(".pth",".csv"))
        results = {col_name:list(results[col_name].values) for col_name in results.columns}
        best_valid_loss = torch.tensor(results['valid_loss'][-1],dtype = torch.float64)
        stp = 1 + len(results['valid_loss'])
    else:
        print('initialize')
        results = dict(
                train_loss      = [],
                valid_loss      = [],
                epochs          = [],
                learning_rate   = [],)
        best_valid_loss         = torch.from_numpy(np.array(np.inf))
        stp = 1
    loss_func,optimizer = createLossAndOptimizer(autoencoder,learning_rate = lr)
    scheduler   = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.9)
    for idx_epoch in range(n_epochs):
        print('Epoch:', idx_epoch + 1,'LR:', scheduler.get_lr())
#        lr_ = lr * (1/(idx_epoch + 1))
#        loss_func,optimizer = createLossAndOptimizer(autoencoder,learning_rate = lr_)
        # train
        print('training ...')
        train_loss          = train_loop(autoencoder,loss_func,optimizer,dataloader_train,device,stp,idx_epoch)
        scheduler.step()
        # validation
        if idx_epoch > 0:
            encoder = encoder2D(batch_size = batch_size)
            decoder = decoder2D(batch_size = batch_size)
            autoencoder     = nn.Sequential(OrderedDict(
                    [('encoder',encoder),
                     ('decoder',decoder),
                            ]
                    )).to(device)
            autoencoder.load_state_dict(torch.load(saving_name))
            autoencoder.eval()
        print('validating ...')
        valid_loss = validation_loop(autoencoder,loss_func,dataloader_valid,device,idx_epoch)
    
        print(f'epoch {idx_epoch + stp}, validation loss = {valid_loss:.6f}')
        print('determine early stop')
        if valid_loss.cpu().clone().detach().type(torch.float64) < best_valid_loss:
            best_valid_loss = valid_loss.cpu().clone().detach().type(torch.float64)
            torch.save(autoencoder.state_dict(),saving_name)
            print('saving model')
        results['train_loss'].append(train_loss.detach().cpu().numpy())
        results['valid_loss'].append(valid_loss.detach().cpu().numpy())
        results['epochs'].append(idx_epoch + stp)
        results['learning_rate'].append(lr)
        results_to_save = pd.DataFrame(results)
        results_to_save.to_csv(saving_name.replace(".pth",".csv"),index = False)



































