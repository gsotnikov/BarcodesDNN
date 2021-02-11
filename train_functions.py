import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import pairwise_distances
from functools import reduce
import operator

import pandas as pd
import sys
import os

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

seed=26


def make_loaders(seed, dataset, batch_size=32):
    dataset_dict = {'MNIST': ['mnist_data/', datasets.MNIST],
                    'FMNIST': ['fashion_mnist_data/', datasets.FashionMNIST],
                    'CIFAR10': ['cifar10_data/', datasets.CIFAR10],
                    'SVHN': ['svhn_data/', datasets.SVHN],
                    'CIFAR100': ['cifar100_data/', datasets.CIFAR100]}

    torch.manual_seed(seed)
    path, function = dataset_dict[dataset]
    
    if dataset != 'SVHN':
        train_loader = torch.utils.data.DataLoader(function(path, download=True, train=True,
                                                   transform=transforms.Compose([transforms.ToTensor()])),
                                                   batch_size=batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(function(path, download=True, train=False,
                                                   transform=transforms.Compose([transforms.ToTensor()])), 
                                                   batch_size=batch_size, shuffle=True)
    else:
        train_loader = torch.utils.data.DataLoader(function(path, download=True, split='train',
                                                   transform=transforms.Compose([transforms.ToTensor()])),
                                                   batch_size=batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(function(path, download=True, split='test',
                                                   transform=transforms.Compose([transforms.ToTensor()])), 
                                                   batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader



def train_scheduler(net, train_loader, optimizer, scheduler, epoch, device, dataset='CIFAR100'):
    net.train()
        
    train_loss = 0
    for batch_id, (data, label) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False, desc=f'Epoch: {epoch}'):
        optimizer.zero_grad()
        
        pred = net(data.to(device))
        
        if dataset == 'CIFAR100':
            n_classes=100
        else:
            n_classes=10
        
        Preds = torch.cat(pred.split(n_classes, dim = 1))
        Labels = torch.cat([label for _ in range(net.N)])
        
        loss = criterion(Preds.to(device), Labels.to(device)) * net.N
        loss.backward()
        
        optimizer.step()
        optimizer.param_groups[0]['lr'] = scheduler.triangle_scheduler(batch_id, epoch)
        
        train_loss += loss.item()
    
    epoch_loss = train_loss/len(train_loader)

    return epoch_loss


def evaluate(net, train_loader, device, dataset='CIFAR100'):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    
    
    train_loss = 0
    acc = []

    for batch_id, (data, label) in enumerate(train_loader):        
        pred = net(data.to(device))
        
        if dataset == 'CIFAR100':
            n_classes=100
        else:
            n_classes=10
        
        Preds = torch.cat(pred.split(n_classes, dim = 1))
        Labels = torch.cat([label for _ in range(net.N)])

        acc.append(((Preds.cpu().argmax(dim=1).reshape(-1) == Labels.cpu().reshape(-1))).float().mean().item())
        
        loss = criterion(Preds.to(device), Labels.to(device)) * net.N
        
        train_loss += loss.item()
    
    net.train()
    
    return train_loss/len(train_loader), np.mean(acc)


def evaluate_single(net, train_loader, device):
    criterion = nn.CrossEntropyLoss()

    train_loss = 0
    for batch_id, (data, label) in enumerate(train_loader):        
        pred = net(data.to(device))
        loss = criterion(pred, label.to(device))
        train_loss += loss.item()

    return train_loss/len(train_loader)