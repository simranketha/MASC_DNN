#!/usr/bin/env python
# coding: utf-8

# In[36]:


# CODE FOR nearest class mean with cosine.. 

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
import torch.nn.functional as F


# In[19]:


def class_mean_fn(train_x,train_y):
    num_classes = train_y.max().item() + 1
    feature_dim = train_x.size(1)
    class_means = torch.zeros(num_classes, feature_dim)

    for c in range(num_classes):
        class_mask = (train_y == c)         
        class_vectors = train_x[class_mask]  
        class_mean = class_vectors.mean(dim=0)  
        class_means[c] = class_mean

    return class_means

def nearest_class_mean_fn(X_test,class_means,test_y):

    X_test_norm = F.normalize(X_test, p=2, dim=1)        # shape: (num_test_samples, 128)
    class_means_norm = F.normalize(class_means, p=2, dim=1)  # shape: (num_classes, 128)

    # Compute cosine similarity: (num_test_samples, num_classes)
    cos_sim = torch.matmul(X_test_norm, class_means_norm.T)

    # For each test sample, pick the class with highest cosine similarity
    pred_y = torch.argmax(cos_sim, dim=1)
    
    accuracy = (pred_y == test_y).float().mean()
    print(f"Accuracy: {accuracy.item() * 100:.2f}%")
    return accuracy.item()

def compute_flops(train_x, train_y, test_x, num_classes):
    N_train, D = train_x.shape
    N_test = test_x.shape[0]

    flops_train = D * N_train + D * num_classes
    flops_infer = N_test * (2*D + D*num_classes + num_classes)
    return flops_train, flops_infer

def probe_nearest_class_mean(type_network,run,results,epoch_present,
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
        t0 = time.time()
        class_means=class_mean_fn(train_x,train_y)
        class_means=class_means.to(dev)
        t1 = time.time()
        total1 = t1-t0
        print(f'total time taken {total1} sec for training_probe')

        accuracy=nearest_class_mean_fn(test_x,class_means,original_testy)

        t2 = time.time()
        total2 = t2-t1
        print(f'total time taken {total2} sec for inference')
        
        flops_train, flops_infer = compute_flops(train_x, train_y, test_x, class_means.size(0))
        
        filename='acc_overall_test'
        d = {'acc_overall':[accuracy],
            'training_probe time':[total1],
            'inference time':[total2],
             'training_probe flops':[flops_train],
             'inference flops':[flops_infer]
            }
        df1 = pd.DataFrame(data=d)
        df1.to_csv(f"{path2}/layer_{p_layer}_{filename}.csv") 


# In[9]:


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description="Select model_type, datasets and corruption.")
    
    corrution_prob = [0.0,0.2,0.4,0.6,0.8,1.0]
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

#     # Access arguments
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

    runs = 4
    # for corrupt in corrution_prob:

    results_folder=f'results/nearest_class_mean/{ds}_{type_network}'
    results_corr,results_org=loading_saving_data.results_fol_name(results_folder,
                                                                  corrupt)

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

    probe_nearest_class_mean(type_network,run,results_corr,
                             epoch_present,subspace_type='corrupt',dev=dev)


    print('original subspace')
    probe_nearest_class_mean(type_network,run,results_org,
                             epoch_present,subspace_type='original',dev=dev)
    print(f"run {run} done")

    shutil.rmtree(temp_path)
    print(f'corrupt {corrupt} done')

