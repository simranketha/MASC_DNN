#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

from MASC import cnn_create
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

############### corrupted data loader and labels
def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', 'yes', '1'}:
        return True
    elif value.lower() in {'false', 'no', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




def traget_train(ds):
    if ds=='CIFAR10':
        max_value=100
    if ((ds=='TinyImageNet') or (ds=='MNIST')):
        max_value=99.9
    if ((ds=='FashionMNIST') or (ds=='CIFAR100')):
        max_value=99.1
    return max_value



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select model_type, datasets,run and corruption.")

    corrution_prob =[0.0,0.2,0.4,0.6,0.8,1.0] 
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
        
    # corrupt = 0.0
    type_network = 'ResNet18'
    ds = 'CIFAR10'
    dropout=False

    torch.manual_seed(42)

    runs = 4

    test_loader,batch_size=cnn_create.test_loading(ds)

    #folder set up for training epochs
    path,network_path,res_path=cnn_create.path_model(ds,type_network,dropout=dropout)

    # for corrupt in corrution_prob:

    cur_network_dir = os.path.join(network_path, str(corrupt))
    os.makedirs(cur_network_dir, exist_ok=True)
    corrupted_train, og_targets, cor_targets=cnn_create.train_loading(ds,batch_size,corrupt)

    #         for run in range(1,runs):
    run_path = os.path.join(cur_network_dir,f'Run_{run}')
    os.makedirs(run_path,exist_ok=True)

    results = {
                'epoch': [],
                'Train_accuracy': [],
                'Test_accuracy': [],
            }
    model = cnn_create.ResNet18()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.to(device)

    print(f"code is running on device: {device}")

    torch.save(model.state_dict(),os.path.join(run_path,'./initialized_model.pth'))

    epoches = 500

    for epoch in range(epoches):
        train_loss = 0
        test_loss = 0
        train_acc = 0
        test_acc = 0
        temp_acc = 0
        count = 0

        for idx, (inputs, labels) in tqdm(enumerate(corrupted_train), 
                                          total=len(corrupted_train), 
                                          desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()
            output = model(inputs)
            loss = loss_func(output, labels)
            loss.backward()
            train_loss += loss.item()
            temp_acc += (torch.argmax(output, 1) == labels).float().sum().item()
            count += len(labels)
            optimizer.step()
        train_acc = (temp_acc/count) * 100

        print(f'\n Corrupt: {corrupt}, Run: {run} Epoch: {epoch} Training accuracy:{train_acc}, correct: {temp_acc}, total:{count}')
        results['Train_accuracy'].append(train_acc)
        results['epoch'].append(epoch+1)

        acc = 0
        test_count = 0
        total_loss = 0
        batches = 0
        model.eval()
        with torch.no_grad():
            for i, (x_batch, y_batch) in tqdm(enumerate(test_loader), 
                                              total=len(test_loader), 
                                              desc="Testing Round"):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(x_batch)
                loss = loss_func(y_pred, y_batch)
                total_loss += loss.item()
                acc += (torch.argmax(y_pred, 1) == y_batch).float().sum().item()
                test_count += len(y_batch)
                batches = i + 1

        test_acc = (acc / test_count) * 100
        results['Test_accuracy'].append(test_acc)
        model.train()
        print(f'\n Corrupt: {corrupt}, Run: {run} Epoch: {epoch} Testing accuracy:{test_acc}, correct: {acc}, total:{test_count}')
        torch.save(model.to('cpu').state_dict(), os.path.join(run_path,f'model_{epoch}.pth'))

        model.to(device)

        if ((train_acc >= traget_train(ds)) or (train_acc == 100.0)):
            break

        

    temp_res_path = os.path.join(res_path,str(corrupt))
    os.makedirs(temp_res_path, exist_ok=True)
    df = pd.DataFrame(data=results)
    df.to_csv(os.path.join(temp_res_path,f"Run_{run}.csv"))
    del corrupted_train


