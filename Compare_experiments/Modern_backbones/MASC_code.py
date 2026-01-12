#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

import pandas as pd

from tqdm import tqdm
import copy
import argparse
import time
torch.multiprocessing.set_sharing_strategy('file_system')

from MASC import cnn_create
from MASC import angle_pytorch as angle

import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.multiprocessing.set_sharing_strategy('file_system')
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import shutil


# In[2]:


def temp_store(ds,type_network,corrupt,run,n_value,seed_value):
    #path for temprary activation storage
    temp_path = f'/mnt/SSD1TB/Network_data/{ds}_{type_network}_{corrupt}_{run}_{seed_value}'
    os.makedirs(temp_path,exist_ok=True)
      
    results_folder=f'results/{seed_value}/angle_results_{n_value}/{ds}_{type_network}'
    os.makedirs(results_folder,exist_ok=True)
    
    #results folders for corrupted subspace
    results_corr={}
    file_name=f'{results_folder}/results_{corrupt}/angle_results'
    os.makedirs(file_name,exist_ok=True)
    results_corr['angle_1']=file_name
    
    results_org={}
    file_name=f'{results_folder}/results_{corrupt}/angle_results_exp3'
    os.makedirs(file_name,exist_ok=True)
    results_org['angle_1']=file_name
    

    #results pca
    results_folder=f'/mnt/2TB/simran/MASC/pca_saved/TMLR_compare/{seed_value}/angle_results_{n_value}/{ds}_{type_network}'
    file_name=f'{results_folder}/results_{corrupt}/pca_corrupted'
    os.makedirs(file_name,exist_ok=True)
    results_corr['pca']=file_name
    
    file_name=f'{results_folder}/results_{corrupt}/pca_original'
    os.makedirs(file_name,exist_ok=True)
    results_org['pca']=file_name
    
    return temp_path,results_corr,results_org


# In[3]:



def test_loading_batch(ds):
    batch_size=1
    if ds=='MNIST':
        _, test_loader = cnn_create.load_data("test",num_workers=1,batch_size=batch_size,mnist=True)
        
    if ds=='FashionMNIST':
        _, test_loader = cnn_create.load_data("test",num_workers=1,batch_size=batch_size,fashion=True)
        
    if ds=='CIFAR10':
        _, test_loader = cnn_create.load_data("test",num_workers=1,batch_size=batch_size,cifar10=True)
        
    if ds=='CIFAR100':
        _, test_loader = cnn_create.load_data("test",num_workers=1,batch_size=batch_size)
        
    if ds=='TinyImageNet':
        
        _, test_loader , _, _, _ = cnn_create.get_cifar_dataloaders_corrupted(batch_size=batch_size,
                                                                   tiny_imagenet=True)
    
    return test_loader



def epoch_number(network_path,corrupt,run):
    models=sorted(os.listdir(f'{network_path}/{corrupt}/Run_{run}'))
    model_numbers = [
        (m, int(m.split("_")[1].split(".")[0]))  
        for m in models if m.startswith("model_") and m.split("_")[1].split(".")[0].isdigit()
    ]
    model_highest = max(model_numbers, key=lambda x: x[1])
    return model_highest[1]




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select model_type, dataset, corrupt, run, n_value.")

    corrution_prob =[0.0,0.2,0.4,0.6,0.8] 
    run_values=[1,2,3]

    parser.add_argument(
        "-corr", type=float, required=True, choices=corrution_prob, help="select corruption"
    )

    parser.add_argument(
        "-run", type=int, required=True, choices=run_values, help="select run"
    )

    args = parser.parse_args()
    corrupt = args.corr
    run=args.run
    
    if corrupt not in corrution_prob:
        args.print_help()
    if run not in run_values:
        args.print_help()
    
    type_network = 'ResNet18'
    ds ='CIFAR10'

    n_value=0.99

    runs=4
    seed_value=40

#     for corrupt in corrution_prob:

    torch.manual_seed(seed_value)
    network_path=cnn_create.path_network_fn(type_network,ds)

    for run in range(1,runs):
        print(ds,type_network,run)
        temp_path,results_corr,results_org=temp_store(ds,type_network,
                                          corrupt,run,n_value,seed_value)

        epoch=epoch_number(network_path,corrupt,run)

        angle.masc_probe(type_network,ds,temp_path,run,
                         results_corr,n_value,epoch,dev,subspace_type='corrupt')
        angle.masc_probe(type_network,ds,temp_path,run,
                         results_org,n_value,epoch,dev,subspace_type='original')

    #     shutil.rmtree(temp_path)
    #     torch.cuda.empty_cache()

        print(f'run : {run} corrupt : {corrupt} seed : {seed_value} done')





