#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 12:04:29 2021

@author: nmei
"""
import os

import numpy as np

from glob import glob
from tqdm import tqdm
from nibabel import load as load_fmri

import torch
from torch          import nn,no_grad
from torch.utils    import data
from torch.nn       import functional as F
from torch          import optim
from torch.autograd import Variable

from torchvision.datasets import ImageFolder
from torchvision          import transforms
from torchvision          import models as Tmodels

#candidate models
def candidates(model_name,pretrained = True,):
    picked_models = dict(
            resnet18        = Tmodels.resnet18(pretrained           = pretrained,
                                              progress              = False,),
            alexnet         = Tmodels.alexnet(pretrained            = pretrained,
                                             progress               = False,),
            # squeezenet      = Tmodels.squeezenet1_1(pretrained      = pretrained,
            #                                        progress         = False,),
            vgg19           = Tmodels.vgg19_bn(pretrained           = pretrained,
                                              progress              = False,),
            densenet169     = Tmodels.densenet169(pretrained        = pretrained,
                                                 progress           = False,),
            inception       = Tmodels.inception_v3(pretrained       = pretrained,
                                                  progress          = False,),
            # googlenet       = Tmodels.googlenet(pretrained          = pretrained,
            #                                    progress             = False,),
            # shufflenet      = Tmodels.shufflenet_v2_x0_5(pretrained = pretrained,
            #                                             progress    = False,),
            mobilenet       = Tmodels.mobilenet_v2(pretrained       = pretrained,
                                                  progress          = False,),
            # resnext50_32x4d = Tmodels.resnext50_32x4d(pretrained    = pretrained,
            #                                          progress       = False,),
            resnet50        = Tmodels.resnet50(pretrained           = pretrained,
                                              progress              = False,),
            )
    return picked_models[model_name]

def define_type(model_name):
    model_type          = dict(
            alexnet     = 'simple',
            vgg19       = 'simple',
            densenet169 = 'simple',
            inception   = 'inception',
            mobilenet   = 'simple',
            resnet18    = 'resnet',
            resnet50    = 'resnet',
            )
    return model_type[model_name]

def hidden_activation_functions(activation_func_name):
    funcs = dict(relu = nn.ReLU(),
                 selu = nn.SELU(),
                 elu = nn.ELU(),
                 sigmoid = nn.Sigmoid(),
                 tanh = nn.Tanh(),
                 linear = None,
                 )
    return funcs[activation_func_name]

class customizedDataset(data.Dataset):
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

##############################################################################
class Simple2DEncoder(nn.Module):
    def __init__(self,
                 pretrained_model_name,
                 batch_size         = 10,
                 device             = 'cpu',
                 in_shape           = (1,66,88,88),
                 hidden_units       = 128,
                 pretrained         = True,
                 ):
        super(Simple2DEncoder, self).__init__()
        
        self.batch_size = batch_size
        self.device     = device
        self.pretrained = pretrained
        torch.manual_seed(12345)
        if define_type(pretrained_model_name) == 'simple':
            pretrained_model    = candidates(pretrained_model_name,pretrained = pretrained)
            layer = pretrained_model.features[0][0]
            copy_weights = 0
            # create new first layer
            new_layer = nn.Conv2d(in_channels = in_shape[1], 
                                  out_channels = layer.out_channels,
                                  kernel_size = layer.kernel_size,
                                  stride = layer.stride,
                                  padding = layer.padding,
                                  bias = layer.bias,
                                  )
            new_layer.weight[:,:layer.in_channels,:,:] = layer.weight.clone()
            for ii in range(in_shape[1] - layer.in_channels):
                channel = layer.in_channels + 1
                new_layer.weight[:,channel:channel+1,:,:] = layer.weight[:,copy_weights:copy_weights + 1,:,:].clone()
            new_layer.weight = nn.Parameter(new_layer.weight)
            pretrained_model.features[0][0] = new_layer
        # freeze the weights of the CNN
        if self.pretrained:
            for layer in pretrained_model.parameters():
                layer.requires_grad = False
        adaptive_pooling        = nn.AdaptiveAvgPool2d((1,1))
        in_features             = nn.AdaptiveAvgPool2d((1,1))(pretrained_model.features(torch.rand(*in_shape))).shape[1]
        if self.pretrained:
            self.hidden_layer_1 = nn.Linear(in_features,hidden_units).to(device)
            self.hidden_layer_2 = nn.Linear(in_features,hidden_units).to(device)
        
        print(f'feature dim = {in_features}')
        self.features           = nn.Sequential(pretrained_model.features,
                                                adaptive_pooling,).to(device)
    def forward(self,x):
        # transpose
        x1 = x.permute([0,3,1,2])
        x2 = x.permute([0,3,2,1])
        # extract representations
        if self.pretrained:
            with torch.no_grad():
                rep1 = torch.squeeze(torch.squeeze(self.features(x1),3),2)
                rep2 = torch.squeeze(torch.squeeze(self.features(x2),3),2)
        else:
            rep1 = torch.squeeze(torch.squeeze(self.features(x1),3),2)
            rep2 = torch.squeeze(torch.squeeze(self.features(x2),3),2)
        if self.pretrained:
            # extra layer
            rep1 = torch.sigmoid(self.hidden_layer_1(rep1))
            rep2 = torch.sigmoid(self.hidden_layer_2(rep2))
        # normalize for cosine distance
        rep1 = rep1 - rep1.mean(1).reshape(-1,1)
        rep2 = rep2 - rep2.mean(1).reshape(-1,1)
        
        return rep1,rep2
##############################################################################

def cosine2by2(representation1, representation2,device = 'cpu'):
    loss_func = nn.CosineEmbeddingLoss()
    x1 = torch.cat([representation1,representation1])
    x2 = torch.cat([representation2,torch.flipud(representation2)])
    y = torch.tensor([1,1,-1,-1],)
    idx = np.random.choice(len(y),len(y),replace = False)
    return loss_func(x1[idx].to(device),x2[idx].to(device),y[idx].to(device))

def train_loop(
        net,
        dataloader,
        optimizer,
        # loss_func,
        device,
        idx_epoch = 0,
        print_train = True,
        ):
    train_loss = 0.
    # set the model to "train"
    net.to(device).train(True)
    # verbose level
    iterator   = tqdm(enumerate(dataloader)) if print_train else enumerate(dataloader)
    
    for ii,(input_volume,_,_,file_name) in iterator:
        # first thing to do is to zero grad the optimizer for the current batch
        optimizer.zero_grad()
        # forward pass
        input_volume = Variable(input_volume).to(device)
        rep1,rep2 = net(input_volume)
        # calculate the loss
        loss_batch = cosine2by2(rep1,rep2,device = device,)
        # backpropagation
        loss_batch.backward()
        # modify the weights
        optimizer.step()
        # record the training loss of a mini-batch
        train_loss  += loss_batch.data
        if print_train:
            iterator.set_description(f'epoch {idx_epoch+1}-{ii + 1:3.0f}/{100*(ii+1)/len(dataloader):2.3f}%,loss = {train_loss/(ii+1):.6f}')
    
    return train_loss

def valid_loop(
        net,
        dataloader,
        # loss_func,
        device,
        idx_epoch = 0,
        print_train = True,):
    valid_loss = 0.
    # set the model to "train"
    net.to(device).eval()
    # verbose level
    iterator   = tqdm(enumerate(dataloader)) if print_train else enumerate(dataloader)
    
    for ii,(input_volume,_,_,file_name) in iterator:
        with torch.no_grad():
            # forward pass
            input_volume = Variable(input_volume).to(device)
            rep1,rep2 = net(input_volume)
            # calculate the loss
            loss_batch = cosine2by2(rep1,rep2,device = device,)
            valid_loss  += loss_batch.data
        if print_train:
            iterator.set_description(f'epoch {idx_epoch+1}-{ii + 1:3.0f}/{100*(ii+1)/len(dataloader):2.3f}%,loss = {valid_loss/(ii+1):.6f}')
    
    return valid_loss
















