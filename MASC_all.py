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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def temp_store(ds,type_network,corrupt,run):
    #path for temprary activation storage
    temp_path = 'Network_data/{ds}_{type_network}_{corrupt}_{run}'
    os.makedirs(temp_path,exist_ok=True)
      
    results_folder=f'angle_results/{ds}_{type_network}'
    os.makedirs(results_folder,exist_ok=True)
    
    #results foldersfor corrupted subspace
    results_corr={}
    os.makedirs(f'{results_folder}/results_{corrupt}',exist_ok=True)
    os.makedirs(f'{results_folder}/results_{corrupt}/angle_results',exist_ok=True)
    results_corr['angle_1']=f'{results_folder}/results_{corrupt}/angle_results'
    #original training labels + corrupted training subspaces : exp2
    os.makedirs(f'{results_folder}/results_{corrupt}/angle_results_exp2',exist_ok=True)
    results_corr['angle_2']=f'{results_folder}/results_{corrupt}/angle_results_exp2'
    #results pca
    os.makedirs(f'{results_folder}/results_{corrupt}/pca_corrupted',exist_ok=True)
    results_corr['pca']=f'{results_folder}/results_{corrupt}/pca_corrupted'  
    #results folders for original subspace
    results_org={}
    os.makedirs(f'{results_folder}/results_{corrupt}/angle_results_exp3',exist_ok=True)
    results_org['angle_1']=f'{results_folder}/results_{corrupt}/angle_results_exp3'
    results_org['pca']=f'{results_folder}/results_{corrupt}/pca_original'  
    
    
    return temp_path,results_corr,results_org

def temp_store_exp3(ds,type_network,corrupt,run):
    #path for temprary activation storage
    temp_path = 'Network_data_exp3/{ds}_{type_network}_{corrupt}_{run}'
    os.makedirs(temp_path,exist_ok=True)
      
    results_folder=f'angle_results/{ds}_{type_network}'
    os.makedirs(results_folder,exist_ok=True)
    
    #results foldersfor corrupted subspace
    results_corr={}
    os.makedirs(f'{results_folder}/results_uncorrupted/results_{corrupt}',exist_ok=True)
    os.makedirs(f'{results_folder}/results_uncorrupted/results_{corrupt}/angle_results',exist_ok=True)
    results_corr['angle_1']=f'{results_folder}/results_uncorrupted/results_{corrupt}/angle_results'
    #original training labels + corrupted training subspaces : exp2
    os.makedirs(f'{results_folder}/results_uncorrupted/results_{corrupt}/angle_results_exp2',exist_ok=True)
    results_corr['angle_2']=f'{results_folder}/results_uncorrupted/results_{corrupt}/angle_results_exp2'
    
    #results pca
    os.makedirs(f'{results_folder}/results_uncorrupted/results_{corrupt}/PCA',exist_ok=True)
    results_corr['pca']=f'{results_folder}/results_uncorrupted/results_{corrupt}/PCA'  

    
    return temp_path,results_corr

def test_loading_batch(ds):
    batch_size=1
    if ds=='MNIST':
        _, test_loader = load_data("test",num_workers=1,batch_size=batch_size,mnist=True)
        
    if ds=='FashionMNIST':
        _, test_loader = load_data("test",num_workers=1,batch_size=batch_size,fashion=True)
        
    if ds=='CIFAR10':
        _, test_loader = load_data("test",num_workers=1,batch_size=batch_size,cifar10=True)
        
    if ds=='CIFAR100':
        _, test_loader = load_data("test",num_workers=1,batch_size=batch_size)
        
    if ds=='TinyImageNet':
        
        _, test_loader , _, _, _ = get_cifar_dataloaders_corrupted(batch_size=batch_size,
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

"""**layer_output**"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select model_type, datasets, run, dropout, exp  and corruption.")

    corrution_prob =[0.0,0.2,0.4,0.6,0.8,1.0] 
    model_type = ['CNN','AlexNet']
    datasets = ['CIFAR10','MNIST','FashionMNIST','TinyImageNet','CIFAR100']
    run_values=[1,2,3]
    bool_values=['True','False']


    parser.add_argument(
        "-corr", type=float, required=True, choices=corrution_prob, help="select corruption"
    )
    parser.add_argument(
        "-model", type=str, required=True, choices=model_type, help="select model_type"
    )
    parser.add_argument(
        "-dataset", type=str, required=True, choices=datasets, help="select dataset"
    )
    parser.add_argument(
        "-run", type=int, required=True, choices=drop_out_values, help="select run"
    )
    
    parser.add_argument(
        "-dropout", type=str, choices=bool_values, help="select dropout flag"
    )
    
    parser.add_argument(
        "-exp3", type=str, choices=bool_values, help="select exp3 flag"
    )
    
    parser.add_argument(
        "-exp2", type=str, choices=bool_values, help="select exp2 flag"
    )
    parser.add_argument(
        "-exp1", type=str, choices=bool_values, help="select exp1 flag"
    )

    args = parser.parse_args()
    corrupt = args.corr
    type_network = args.model
    ds = args.dataset
    run=args.run
    dropout=str2bool(args.dropout)
    if corrupt not in corrution_prob:
        args.print_help()
    if type_network not in model_type:
        args.print_help()
    if ds not in datasets:
        args.print_help()
    if run not in run_values:
        args.print_help()
    if dropout not in bool_values:
        args.print_help()
    if exp3 not in bool_values:
        args.print_help()  
    if exp2 not in bool_values:
        args.print_help() 
    if exp1 not in bool_values:
        args.print_help() 
    torch.manual_seed(42)
        
    n=0.99
    runs = 4
    #getting the path for the model and path for temp storage
    path,network_path,_=cnn_create.path_model(ds,dropout=dropout)
    if exp3:
        temp_path,results_corr=temp_store_exp3(ds,type_network,corrupt,run)
        model_corrupt=0.0
    if (exp1 or exp2):
        temp_path,results_corr,results_org=temp_store(ds,type_network,corrupt,run)
        model_corrupt=corrupt
        
    test_loader=test_loading_batch(ds)

#     for corrupt in corrution_prob:

    corrupted_train, og_targets, cor_targets=cnn_create.train_loading(ds,batch_size=1,corrupt)

#     for run in range(1,runs):

    epoch=epoch_number(network_path,corrupt_model,run)

    dummy_model,_,_ = cnn_create.model_build(type_network,ds,dropout=dropout)
    path_file_load=f'{network_path}/{model_corrupt}/Run_{run}/model_{epoch}.pth'
    dummy_model.load_state_dict(torch.load(path_file_load))


    cnn_create.loading_saving_activations(temp_path,dummy_model,corrupted_train,
                               test_loader,og_targets,dev,type_network,dropout=dropout)
    
    if exp1 or exp3:
        angle.angle_work_corrupt(type_network,temp_path,run,results_corr,n,dropout=dropout)   
    
    if exp2:
        angle.angle_work_original(type_network,temp_path,run,results_org,n,dropout=dropout)   
   
    print('deleting temp folder')
    for file in os.listdir(temp_path):                 
        os.remove(f'{temp_path}/{file}')
    os.rmdir(temp_path)
    print(f'run {run} done')
    print(f'corrupt {corrupt} done')
    





