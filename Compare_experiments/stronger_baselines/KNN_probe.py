#!/usr/bin/env python
# coding: utf-8

# In[1]:


# CODE FOR KNN in feature space (1d)

from CNN_code import cnn_create
from probe import loading_saving_data

import os
import torch
import time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.multiprocessing.set_sharing_strategy('file_system')
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import pandas as pd
import argparse
import shutil

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np


# In[2]:


def knn_prob(train_x,train_y,test_x,original_testy):
    train_x = train_x.cpu().numpy()
    train_y = train_y.cpu().numpy()     
    test_x = test_x.cpu().numpy()
    original_testy = original_testy.cpu().numpy()

    # --- knn ---
    t0 = time.time()
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')  # you can use 'euclidean' too

    knn.fit(train_x, train_y)
    t1 = time.time()
    total1 = t1-t0
    print(f'total time taken {total1} sec for training_probe')
    y_pred = knn.predict(test_x)

    acc_knn = accuracy_score(original_testy, y_pred)
    print(f"KNN Accuracy: {acc_knn*100:.2f}%")   

    t2 = time.time()
    total2 = t2-t1
    print(f'total time taken {total2} sec for inference') 
    
    
    
    # --- Analytical FLOPs ---
    n_train, n_features = train_x.shape
    n_test = test_x.shape[0]

    # Metric constant: cosine uses ~4 FLOPs per feature, euclidean ~2
    metric = 'cosine'
    c = 4 if metric == 'cosine' else 2

    # Training FLOPs (negligible)
    flops_train = 0

    # Inference FLOPs (distance computations)
    flops_infer = n_test * n_train * c * n_features

    print(f"Estimated Training FLOPs: {flops_train/1e9:.6f} GFLOPs (negligible)")
    print(f"Estimated Inference FLOPs: {flops_infer/1e9:.3f} GFLOPs")

    return acc_knn, total1, total2, flops_train, flops_infer


# In[4]:



def probe_KNN(type_network,run,results,epoch_present,
              subspace_type,dev):
    print(type_network)
    pca_layer,data_layer,_=loading_saving_data.layer_name(type_network)
    results_angle1=results['angle_1'] 
    epoch=epoch_present

    #results folders 
    path2=f'{results_angle1}/Run_{run}/{epoch}'
    os.makedirs(path2,exist_ok=True)

    for p_layer,d_layer in zip(pca_layer,data_layer):
        print(p_layer,d_layer)
        if subspace_type=='corrupt':
            train_x,_,train_y=loading_saving_data.data_loading_train(temp_path,d_layer)
        else:
            train_x,train_y,_=loading_saving_data.data_loading_train(temp_path,d_layer)    
        test_x,original_testy=loading_saving_data.data_loading_test(temp_path,d_layer)

        acc_knn,total1,total2,flops_train,flops_infer=knn_prob(train_x,train_y,test_x,original_testy)
        flops_train=flops_train/1e9
        flops_infer=flops_infer/1e9
        
        filename='acc_overall_test'
        d = {'acc_overall':[acc_knn],
            'training_probe time':[total1],
            'inference time':[total2],
             'training_probe flops':[flops_train],
             'inference flops':[flops_infer]}
        df1 = pd.DataFrame(data=d)
        df1.to_csv(f"{path2}/layer_{p_layer}_{filename}.csv") 


# In[ ]:


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description="Select model_type, datasets and corruption.")
    
    corrution_prob = [0.0,0.2,0.4,0.6,0.8]#,1.0
    model_type = ['CNN','MLP','AlexNet']
    datasets = ['CIFAR10','MNIST','FashionMNIST','TinyImageNet','CIFAR100']
    run_values=[1,2,3]

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
        "-run", type=int, required=True, choices=run_values, help="select run"
    )

    args = parser.parse_args()

    # Access arguments
    corrupt = args.corr
    type_network = args.model
    ds = args.dataset
    run=args.run
    
    if corrupt not in corrution_prob:
        args.print_help()
    if type_network not in model_type:
        args.print_help()
    if ds not in datasets:
        args.print_help()
    if run not in run_values:
        args.print_help()
    torch.manual_seed(42)

    print(type_network,ds)

    runs = 4
#     for corrupt in corrution_prob:
    print(corrupt)
    results_folder=f'results/KNN/{ds}_{type_network}'
    results_corr,results_org=loading_saving_data.results_fol_name(results_folder,corrupt)

    network_path,test_loader,corrupted_train,og_targets=cnn_create.original_dataset(type_network,ds,corrupt)

#     for run in range(1,runs): 

    #path for temprary activation storage
    temp_path = f'Network_data_{corrupt}_{ds}_{type_network}_{run}'
    os.makedirs(temp_path,exist_ok=True)

    epoch=loading_saving_data.trained_model(network_path,corrupt,run)

    print(epoch)
    dummy_model=cnn_create.model_create(type_network,ds)

    dummy_model.load_state_dict(torch.load(f'{network_path}/{corrupt}/Run_{run}/{epoch}'))

    cnn_create.loading_saving_activations(temp_path,dummy_model,
                                          corrupted_train,test_loader, 
                                          og_targets,dev,type_network)
    del dummy_model

    epoch_present=cnn_create.epochnumber(epoch)

    print('corrupt subspace')
    probe_KNN(type_network,run,results_corr,epoch_present,subspace_type='corrupt',dev=dev)

    print('original subspace')
    probe_KNN(type_network,run,results_org,epoch_present,subspace_type='original',dev=dev)
    print(f"run {run} done")

    shutil.rmtree(temp_path)
    print(f'corrupt {corrupt} done')

