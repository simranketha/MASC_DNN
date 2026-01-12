#!/usr/bin/env python
# coding: utf-8

# In[27]:



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

import gc
from MASC import subspace_pytorch as subspace
from MASC import scratch_pca as sp


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
    file_name=f'{results_folder}/results_{corrupt}/angle_results_exp2'
    os.makedirs(file_name,exist_ok=True)
    results_corr['angle_2']=file_name
    
    results_org={}
    file_name=f'{results_folder}/results_{corrupt}/angle_results_exp3'
    os.makedirs(file_name,exist_ok=True)
    results_org['angle_1']=file_name
    file_name=f'{results_folder}/results_{corrupt}/angle_results_exp3'
    os.makedirs(file_name,exist_ok=True)
    results_org['angle_2']=file_name
    
    #results pca
#     /mnt/2TB
    results_folder=f'/mnt/SSD1TB/simran/MASC/pca_saved/TMLR_compare/{seed_value}/angle_results_{n_value}/{ds}_{type_network}'
    file_name=f'{results_folder}/results_{corrupt}/pca_corrupted'
    os.makedirs(file_name,exist_ok=True)
    results_corr['pca']=file_name
    
    file_name=f'{results_folder}/results_{corrupt}/pca_original'
    os.makedirs(file_name,exist_ok=True)
    results_org['pca']=file_name
    
    return temp_path,results_corr,results_org


# In[3]:

def epoch_number(network_path,corrupt,run):
    models=sorted(os.listdir(f'{network_path}/{corrupt}/Run_{run}'))
    model_numbers = [
        (m, int(m.split("_")[1].split(".")[0]))  
        for m in models if m.startswith("model_") and m.split("_")[1].split(".")[0].isdigit()
    ]
    model_highest = max(model_numbers, key=lambda x: x[1])
    return model_highest[1]

def layer_names(type_network,ds):
    if type_network =='ResNet18':
        if ds=='CIFAR10':
            
            data_layer_name=['after_layer_0_3',
                         'y_value_corrupted']#after_layer_0,'after_layer_0_1','after_layer_0_2'
#             ,'after_layer_0_3','after_layer_1',
#             'after_layer_2','after_layer_3','after_layer_4','before_fc',
            data_output_name=['l0_3']#'l0','l0_1','l0_2','l0_3','l1','l2','l3','l4','bf_last'
    
            num_class=10

    return data_layer_name,data_output_name,num_class
            
    
    
def masc_project_train_corrupt(temp_path,path2_pca,data,data_out,num_class,dev,label_type):
        #testing images
    with open(f'{temp_path}/{data}_train.pkl', 'rb') as file: 
        myvar = pickle.load(file) 

    layer_output_test = [torch.tensor(np.array(myvar[values])).cuda() for values in range(len(myvar))]
    layer_output_test = torch.stack(layer_output_test).cuda()
    if label_type=='corrupt':
        with open(f'{temp_path}/y_value_corrupted_train.pkl', 'rb') as file: 
            myvar = pickle.load(file)  
    if label_type=='original':
        with open(f'{temp_path}/y_value_original_train.pkl', 'rb') as file: 
            myvar = pickle.load(file)  
            
    original_test_labels = [torch.tensor(np.array(myvar[values])).cuda() for values in range(len(myvar))]

    original_test_labels = torch.stack(original_test_labels).cuda()
    batch_size=100
    flops_per_class = []

    for class_projection in range(0,num_class,1):
        X_ppca=torch.load(f"{path2_pca}/pca_train_corrupt_{data_out}_{class_projection}.pt")       
        X_ppca=X_ppca.to(dev)
#         -----flops

        D, n = X_ppca.shape  # feature dimension and PCA dimension
        num_batches = int(np.ceil(layer_output_test.shape[0] / batch_size))
        total_flops_class = 0
        #         -----flops

        # Project testing data onto classwise subspaces
        for j in range(0, layer_output_test.shape[0], batch_size):
            batch_output = layer_output_test[j:j + batch_size].to(torch.float64)
            X_pca_new = subspace.pca_change_layer(batch_output, X_ppca)
            layer_angle = angle.angle_layer(batch_output, X_pca_new)
            
            #         -----flops 
            B = batch_output.shape[0]        
            flops_proj = 2 * B * D * n 
            flops_angle = 4 * B * D 
            total_flops_class += (flops_proj + flops_angle)

            #         -----flops
            if j == 0:
                class_angle_test_batch = layer_angle
            else:
                class_angle_test_batch = torch.cat((class_angle_test_batch, layer_angle), dim=0) 
            
            
        flops_per_class.append(total_flops_class) #         -----flops
        
        if class_projection == 0:
            class_angle_test = class_angle_test_batch
        else:
            class_angle_test = torch.cat((class_angle_test, class_angle_test_batch), dim=0)   

    #test data
    num_images=layer_output_test.shape[0]
    y_pred=angle.least_class_layer(class_angle_test,num_images,
                             number_class=num_class)
    #true label test data
    acc_overall=angle.accuracy_angle_layer(y_pred,original_test_labels)
    
    total_flops = sum(flops_per_class)

    del layer_output_test,original_test_labels
    gc.collect()
    torch.cuda.empty_cache()
    return acc_overall,total_flops


def masc_probe_train(type_network,ds,temp_path,run,results,n,epoch_present,dev,label_type):

    data_layer_name,data_output_name,num_class=layer_names(type_network,ds)
    
    results_angle1=results['angle_1']
    results_angle2=results['angle_2']

    results_pca=results['pca']
    batch_size=100
    epoch=epoch_present
    #results folders 
    path2=f'{results_angle1}/Run_{run}/{epoch}'
    os.makedirs(path2,exist_ok=True)
    path3=f'{results_angle2}/Run_{run}/{epoch}'
    os.makedirs(path3,exist_ok=True)

    path2_pca=f'{results_pca}/Run_{run}/{epoch}' 
    os.makedirs(path2_pca,exist_ok=True)

    with torch.no_grad(): 
        for data,data_out in zip(data_layer_name,data_output_name):
#             t0 = time.time()

#             flops_train=subspace_creation(temp_path,n,path2_pca,data,data_out,
#                                           num_class,subspace_type)
#             flops_train=flops_train/1e9
            t1 = time.time()
#             total1 = t1-t0
            
            acc_overall,flops_infer=masc_project_train_corrupt(temp_path,path2_pca,data,data_out,
                                                      num_class,dev,label_type)
            flops_infer=flops_infer/1e9
            
            t2 = time.time()
            total2=t2-t1
            filename='acc_overall_train'
            d = {'acc_overall':[acc_overall],
                'inference time':[total2],
                 'inference flops':[flops_infer]
                }
            
            df1 = pd.DataFrame(data=d)
            if label_type=='corrupt':
                df1.to_csv(f"{path2}/layer_{data_out}_{filename}.csv") 
            if label_type=='original':
                df1.to_csv(f"{path3}/layer_{data_out}_{filename}.csv") 
                
            
def masc_project_train_org(temp_path,path2_pca,data,data_out,num_class,dev):
        #testing images
    with open(f'{temp_path}/{data}_train.pkl', 'rb') as file: 
        myvar = pickle.load(file) 

    layer_output_test = [torch.tensor(np.array(myvar[values])).cuda() for values in range(len(myvar))]
    layer_output_test = torch.stack(layer_output_test).cuda()

    with open(f'{temp_path}/y_value_original_train.pkl', 'rb') as file: 
        myvar = pickle.load(file)  
            
    original_test_labels = [torch.tensor(np.array(myvar[values])).cuda() for values in range(len(myvar))]

    original_test_labels = torch.stack(original_test_labels).cuda()
    batch_size=100
    flops_per_class = []

    for class_projection in range(0,num_class,1):
        X_ppca=torch.load(f"{path2_pca}/pca_train_original_{data_out}_{class_projection}.pt")       
        X_ppca=X_ppca.to(dev)
#         -----flops

        D, n = X_ppca.shape  # feature dimension and PCA dimension
        num_batches = int(np.ceil(layer_output_test.shape[0] / batch_size))
        total_flops_class = 0
        #         -----flops

        # Project testing data onto classwise subspaces
        for j in range(0, layer_output_test.shape[0], batch_size):
            batch_output = layer_output_test[j:j + batch_size].to(torch.float64)
            X_pca_new = subspace.pca_change_layer(batch_output, X_ppca)
            layer_angle = angle.angle_layer(batch_output, X_pca_new)
            
            #         -----flops 
            B = batch_output.shape[0]        
            flops_proj = 2 * B * D * n 
            flops_angle = 4 * B * D 
            total_flops_class += (flops_proj + flops_angle)

            #         -----flops
            if j == 0:
                class_angle_test_batch = layer_angle
            else:
                class_angle_test_batch = torch.cat((class_angle_test_batch, layer_angle), dim=0) 
            
            
        flops_per_class.append(total_flops_class) #         -----flops
        
        if class_projection == 0:
            class_angle_test = class_angle_test_batch
        else:
            class_angle_test = torch.cat((class_angle_test, class_angle_test_batch), dim=0)   

    #test data
    num_images=layer_output_test.shape[0]
    y_pred=angle.least_class_layer(class_angle_test,num_images,
                             number_class=num_class)
    #true label test data
    acc_overall=angle.accuracy_angle_layer(y_pred,original_test_labels)
    
    total_flops = sum(flops_per_class)

    del layer_output_test,original_test_labels
    gc.collect()
    torch.cuda.empty_cache()
    return acc_overall,total_flops

def masc_probe_train_ORG(type_network,ds,temp_path,run,results,n,epoch_present,dev):

    data_layer_name,data_output_name,num_class=layer_names(type_network,ds)
    
    results_angle2=results['angle_2']
    results_pca=results['pca']
    batch_size=100
    epoch=epoch_present
    #results folders 

    path3=f'{results_angle2}/Run_{run}/{epoch}'
    os.makedirs(path3,exist_ok=True)

    path2_pca=f'{results_pca}/Run_{run}/{epoch}' 
    os.makedirs(path2_pca,exist_ok=True)

    with torch.no_grad(): 
        for data,data_out in zip(data_layer_name,data_output_name):

            t1 = time.time()
            
            acc_overall,flops_infer=masc_project_train_org(temp_path,path2_pca,
                                                           data,data_out,
                                                      num_class,dev)
            flops_infer=flops_infer/1e9
            
            t2 = time.time()
            total2=t2-t1
            filename='acc_overall_train'
            d = {'acc_overall':[acc_overall],
                'inference time':[total2],
                 'inference flops':[flops_infer]
                }
            
            df1 = pd.DataFrame(data=d)
            
            df1.to_csv(f"{path3}/layer_{data_out}_{filename}.csv")                 
        


# In[28]:



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select model_type, dataset, corrupt, run, n_value.")

    corrution_prob =[0.0,0.2,0.4,0.6,0.8] #
    run_values=[1,2,3]

    parser.add_argument(
        "-corr", type=float, required=True, 
        choices=corrution_prob, help="select corruption"
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

        masc_probe_train(type_network,ds,temp_path,run,results_corr,n_value,
                         epoch,dev,label_type='corrupt')
        masc_probe_train(type_network,ds,temp_path,run,results_corr,n_value,
                         epoch,dev,label_type='original')
        masc_probe_train_ORG(type_network,ds,temp_path,run,results_org,n_value,
                             epoch,dev)

            shutil.rmtree(temp_path)
            torch.cuda.empty_cache()

        print(f'run : {run} corrupt : {corrupt} seed : {seed_value} done')

