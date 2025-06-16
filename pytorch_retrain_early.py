#!/usr/bin/env python
# coding: utf-8

# In[2]:


from MASC import angle_pytorch as angle
from MASC import cnn_create

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import statistics
import copy
import pickle
import argparse
import time
import torch
import copy
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.multiprocessing.set_sharing_strategy('file_system')
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import gc
from sklearn.model_selection import train_test_split 
from torch.utils.data import TensorDataset, DataLoader,Subset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms

import shutil

def originial_data(type_network,ds):
    
    if type_network=='AlexNet':
        if ds=='TinyImageNet':
            network_path = 'models/TinyImagenet_AlexNet/Network'
            retrain_path='models/TinyImagenet_AlexNet/Network'
            batch_size=500
            _, test_loader , _,_,_= cnn_create.get_cifar_dataloaders_corrupted(
                batch_size=1,tiny_imagenet=True)

            corrupted_train, _ , _, og_targets, _ = cnn_create.get_cifar_dataloaders_corrupted(
                corrupt_prob=corrupt, batch_size=1,tiny_imagenet=True)
        
        if ds=='CIFAR100':
            network_path = 'models/CIFAR100_AlexNet/Network'
            retrain_path='models/CIFAR100_AlexNet/Network'

            batch_size = 128

            _, test_loader , _,_,_= cnn_create.get_cifar_dataloaders_corrupted(
                batch_size=1)

            corrupted_train, _ , _, og_targets, _ = cnn_create.get_cifar_dataloaders_corrupted(corrupt_prob=corrupt, 
                                        batch_size=1) 
     
    if type_network =='CNN':
        
        if ds=='FashionMNIST':
            batch_size = 128
            network_path = 'models/FashionMNIST_CNN/Network'
            retrain_path='models/FashionMNIST_CNN/Network'
            _, test_loader , _,_,_= cnn_create.get_cifar_dataloaders_corrupted(
                batch_size=1,fashion=True)

            corrupted_train, _ , _, og_targets, _ = cnn_create.get_cifar_dataloaders_corrupted(corrupt_prob=corrupt, 
                                        batch_size=1,fashion=True)
        if ds=='MNIST':
            batch_size = 128
            network_path = 'models/MNIST_CNN/Network'
            retrain_path='models/MNIST_CNN/Network'
            _, test_loader , _,_,_= cnn_create.get_cifar_dataloaders_corrupted(
                batch_size=1,mnist=True)

            corrupted_train, _ , _, og_targets, _ = cnn_create.get_cifar_dataloaders_corrupted(corrupt_prob=corrupt, 
                                        batch_size=1,mnist=True)    
        if ds=='CIFAR10':
            batch_size = 32
            network_path = 'models/CIFAR10_CNN/Network'
            retrain_path='models/CIFAR10_CNN/Network'
            _, test_loader , _,_,_= cnn_create.get_cifar_dataloaders_corrupted(
                batch_size=1,cifar10=True)

            corrupted_train, _ , _, og_targets, _ = cnn_create.get_cifar_dataloaders_corrupted(corrupt_prob=corrupt, 
                                        batch_size=1,cifar10=True)
    
    return corrupted_train,test_loader,og_targets,retrain_path,network_path,batch_size



def model_build(type_network,ds=None):
    if type_network=='CNN':
        if ds=='CIFAR10':
            dummy_model = cnn_create.NgnCnn()
        else:
            dummy_model = cnn_create.NgnCnn(channels=1)
    if type_network=='AlexNet':
        if ds=='TinyImageNet':
            dummy_model = cnn_create.AlexNet(num_classes=200, tiny_imagenet=True)
        if ds=='CIFAR100':
            dummy_model = cnn_create.AlexNet(num_classes=100)
    return dummy_model


def model_params(type_network,ds):
    if type_network=='AlexNet' and ds=='TinyImageNet':
        loss_func=nn.CrossEntropyLoss().to(dev)
        optimizer=optim.Adam(model.parameters(),betas=(0.9, 0.999),lr = 0.0001)
        
    if type_network=='AlexNet' and ds=='CIFAR100':    
        optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
        loss_func = nn.CrossEntropyLoss()

    if type_network=='CNN':
        optimizer = torch.optim.Adam(model.parameters(),lr=0.0002)
        loss_func = nn.CrossEntropyLoss()
    return 

def test_split(test_loader):

    dataset = test_loader.dataset  # Extract the dataset from DataLoader

    # Get dataset indices
    index = np.arange(len(dataset))  

    # Split indices
    test80_index, test20_index = train_test_split(index, test_size=0.20, random_state=2345)

    # Create subsets
    test_80 = torch.utils.data.Subset(dataset, test80_index)
    test_20 = torch.utils.data.Subset(dataset, test20_index)

    # Create new DataLoaders
    test_loader_80 = torch.utils.data.DataLoader(test_80, batch_size=1, shuffle=False)
    test_loader_20 = torch.utils.data.DataLoader(test_20, batch_size=1, shuffle=False)
    
    return test_loader_80,test_loader_20



def relabels_dataset(corrupted_train,y_train,batch_size):
    # Original dataset
    original_loader = copy.deepcopy(corrupted_train)

    # Indices for sub-loader (use the first 50 samples as an example)
    subset_indices = list(range(len(original_loader)))  

    # New labels (must be of the same length as subset_indices)
    new_labels = y_train.round().to(torch.int64).to('cpu')

    # Create the modified dataset and DataLoader
    subset_dataset = SubsetWithNewLabels(original_loader.dataset, subset_indices, new_labels)
    subset_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False)

    return subset_loader




def epoch_number(network_path,corrupt,run):
    models=sorted(os.listdir(f'{network_path}/{corrupt}/Run_{run}'))
    model_numbers = [
        (m, int(m.split("_")[1].split(".")[0]))  
        for m in models if m.startswith("model_") and m.split("_")[1].split(".")[0].isdigit()
    ]
    model_highest = max(model_numbers, key=lambda x: x[1])
    return model_highest[1]


# In[3]:


def model_files(type_network):
    if type_network == 'AlexNet':
        data_layer_name=['flattern','fc1','fc2'] 
     
    if type_network =='CNN':
        data_layer_name=['input','flattern','fc1','fc2','fc3']    
    
    return data_layer_name

def select_layer_idx(type_network,ds,results_corr,results_org,epoch):
    data_layer_name=model_files(type_network)
    results_angle1=results_corr['angle_1']
    results_angle2=results_org['angle_1']
    acc_corr=[]
    acc_org=[]
    for data_layer in data_layer_name:
        file_name=f'{results_angle1}/Run_{run}/{epoch}/layer_{data_layer}_acc_overall_test.csv'
        acc_corr.append(pd.read_csv(file_name)['acc_overall'][0])
        
        file_name=f'{results_angle2}/Run_{run}/{epoch}/layer_{data_layer}_acc_overall_test.csv'
        acc_org.append(pd.read_csv(file_name)['acc_overall'][0])

    layer_highest_corr=np.argmax(acc_corr)
    layer_highest_org=np.argmax(acc_org)
    return layer_highest_corr,layer_highest_org


# In[4]:


# Custom dataset wrapper
class SubsetWithNewLabels(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, new_labels):
        self.dataset = dataset
        self.indices = indices
        self.new_labels = new_labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        image, _ = self.dataset[original_idx]  # Ignore original label
        return image, self.new_labels[idx]


# In[5]:
def test_acc(test_loader_80,
                     model,loss_func,accuracy_path,run,dev):
    results = {
      'Test_accuracy_80': [],
    }
    model.to(dev)
    test_acc = 0

    acc = 0
    test_count = 0
    total_loss = 0
    batches = 0
    model.eval()
    with torch.no_grad():
        for i, (x_batch, y_batch) in tqdm(
            enumerate(test_loader_80), total=len(test_loader_80), desc="Testing 80 Round"):
            x_batch = x_batch.to(dev)
            y_batch = y_batch.to(dev)
            y_pred = model(x_batch)
            loss = loss_func(y_pred, y_batch.type(torch.int64))
            total_loss += loss.item()
            acc += (torch.argmax(y_pred, 1) == y_batch).float().sum().item()

            test_count += len(y_batch)
            batches = i + 1

    test_acc = (acc / test_count) * 100
    results['Test_accuracy_80'].append(test_acc)

    df = pd.DataFrame(data=results)
    df.to_csv(os.path.join(accuracy_path,f"Run_{run}_test80.csv"))
    

def training_process(subset_loader,test_loader_80,test_loader_20,
                     model,loss_func,optimizer,run_path,accuracy_path,run,dev,
                     epoches=10,early_stopping=False):
    
    if early_stopping==True:
        
        results = {
          'epoch': [],
          'Train_accuracy': [],
          'Test_accuracy_80': [],
           'Test_accuracy_20': [],
        }
    
        patience = 3
        best_val_acc = 0.0
        counter = 0
        early_stop = False

    else:
        results = {
          'epoch': [],
          'Train_accuracy': [],
          'Test_accuracy_80': [],
        }

    model.to(dev)
    
    for epoch in range(epoches):
        model.train()
        
        train_loss = 0
        test_loss = 0
        train_acc = 0
        test_acc = 0
        temp_acc = 0
        count = 0

        for idx, (inputs, labels) in tqdm(enumerate(subset_loader), total=len(subset_loader),desc="Training"):
            inputs, labels = inputs.to(dev), labels.to(dev)
            model.zero_grad()
            output = model(inputs)
            loss = loss_func(output, labels.type(torch.int64))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            count += len(labels)

            temp_acc += (torch.argmax(output, 1) == labels).float().sum().item()
        train_acc = (temp_acc/count) * 100

        print(f'\n Corrupt: {corrupt}, Run: {run} Epoch: {epoch},Training accuracy:{train_acc}, correct: {temp_acc}, total:{count}')
        results['Train_accuracy'].append(train_acc)
        results['epoch'].append(epoch+1)
        
        
        
        
        acc = 0
        test_count = 0
        total_loss = 0
        batches = 0
        model.eval()
        with torch.no_grad():
            for i, (x_batch, y_batch) in tqdm(
                enumerate(test_loader_80), total=len(test_loader_80), desc="Testing 80 Round"):
                x_batch = x_batch.to(dev)
                y_batch = y_batch.to(dev)
                y_pred = model(x_batch)
                loss = loss_func(y_pred, y_batch.type(torch.int64))
                total_loss += loss.item()
                acc += (torch.argmax(y_pred, 1) == y_batch).float().sum().item()

                test_count += len(y_batch)
                batches = i + 1

        test_acc = (acc / test_count) * 100
        results['Test_accuracy_80'].append(test_acc)
        print(f'\n Corrupt: {corrupt}, Run: {run} Epoch: {epoch} Testing 80 accuracy:{test_acc}, correct: {acc}, total:{test_count}')
        

        if early_stopping== True:
            acc = 0
            test_count = 0
            val_loss = 0
            batches = 0
            model.eval()
            with torch.no_grad():
                for i, (x_batch, y_batch) in tqdm(
                    enumerate(test_loader_20), total=len(test_loader_20), desc="Testing 20 Round"):
                    x_batch = x_batch.to(dev)
                    y_batch = y_batch.to(dev)
                    y_pred = model(x_batch)
                    loss = loss_func(y_pred, y_batch.type(torch.int64))
                    val_loss += loss.item()
                    acc += (torch.argmax(y_pred, 1) == y_batch).float().sum().item()

                    test_count += len(y_batch)
                    batches = i + 1
                    

            test_acc = (acc / test_count) * 100
            results['Test_accuracy_20'].append(test_acc)
            
            print(f'\n Corrupt: {corrupt}, Run: {run} Epoch: {epoch} Testing 20 accuracy:{test_acc}, correct: {acc}, total:{test_count}')


            val_acc = acc / test_count
            print(f"Epoch {epoch+1}, Validation Accuracy on 20: {val_acc:.4f}")

             # Early stopping logic (based on accuracy)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered!")
                    early_stop = True
                    break
        
        torch.save(model.to('cpu').state_dict(),
                       os.path.join(run_path,f'model_{epoch}.pth'))
        model.to(dev)
        
    df = pd.DataFrame(data=results)
    df.to_csv(os.path.join(accuracy_path,f"Run_{run}.csv"))

    del subset_loader



def indx_label(C,O,R):
    #wrong_label_correctly_relabelled (O != C) & (O == R)

    val_ind_con1_1 = ((O != C) & (O == R)).nonzero(as_tuple=True)[0]
    val_ind_con1_2 = (O != C).nonzero(as_tuple=True)[0]

    #correct_label_wrongly_relabelled (O == C) & (C != R)

    # Get indices satisfying the condition
    val_ind_con2_1 = ((O == C) & (C != R)).nonzero(as_tuple=True)[0]
    val_ind_con2_2 = ((O == C)).nonzero(as_tuple=True)[0]
    
    return val_ind_con1_1,val_ind_con1_2,val_ind_con2_1,val_ind_con2_2


# In[148]:


def percentage_captured(temp_path,y_train_corr,y_train_org,run,results_path):

    #corrupt labels
    with open(f'{temp_path}/y_value_corrupted_train.pkl', 'rb') as file: 
        myvar = pickle.load(file)     

    cor_targets = [torch.tensor(np.array(myvar[values])).cuda() for values in range(len(myvar))]
    cor_targets = torch.stack(cor_targets).cuda()

    #original labels
    with open(f'{temp_path}/y_value_original_train.pkl', 'rb') as file: 
        myvar = pickle.load(file)  

    og_targets = [torch.tensor(np.array(myvar[values])).cuda() for values in range(len(myvar))]
    og_targets = torch.stack(og_targets).cuda()

    C=cor_targets
    O=og_targets
    R_corr=y_train_corr.round().to(torch.int64)
    R_org=y_train_org.round().to(torch.int64)

    #corrupt subspace
    val_ind_con1_1,val_ind_con1_2,val_ind_con2_1,val_ind_con2_2=indx_label(C,O,
                                                                           R_corr)
    #wrong_label_correctly_relabelled 
    per_wlrc_corr=(val_ind_con1_1.shape[0]/val_ind_con1_2.shape[0])*100
    #correct_label_wrongly_relabelled 
    per_clwr_corr=(val_ind_con2_1.shape[0]/val_ind_con2_2.shape[0])*100

    #original subspace
    val_ind_con1_1,val_ind_con1_2,val_ind_con2_1,val_ind_con2_2=indx_label(C,O,
                                                                           R_org)
    #wrong_label_correctly_relabelled 
    per_wlrc_org=(val_ind_con1_1.shape[0]/val_ind_con1_2.shape[0])*100
    #correct_label_wrongly_relabelled 
    per_clwr_org=(val_ind_con2_1.shape[0]/val_ind_con2_2.shape[0])*100
    
#     percentage in correct (corrupt)
    val_ind_1,val_ind_2=indx_incorrect(C,O,R_corr)
    per_incorrect_corr=(val_ind_1.shape[0]-val_ind_2.shape[0])/val_ind_1.shape[0]
    #     percentage in correct (original)
    val_ind_1,val_ind_2=indx_incorrect(C,O,R_org)
    per_incorrect_org=(val_ind_1.shape[0]-val_ind_2.shape[0])/val_ind_1.shape[0]


    filename=f'Run_{run}_percent'
    
    d= {'per_wlrc_corr':[per_wlrc_corr],
       'per_clwr_corr':[per_clwr_corr],
       'per_wlrc_org':[per_wlrc_org],
       'per_clwr_org':[per_clwr_org],
       'per_incorrect_corr':[per_incorrect_corr],
       'per_incorrect_org':[per_incorrect_org]}
    df1 = pd.DataFrame(data=d)
    df1.to_csv(f"{results_path}/{filename}.csv") 


# In[ ]:
def saving_labels(result_path,dummy_model,corrupted_train,
                  og_targets,dev,type_network):

    
    cnn_create.data_saving(result_path,corrupted_train,
                           dummy_model,'train',dev,type_network)
    
    #saving original labels
    with open(f'{result_path}/y_value_original_train.pkl', 'wb') as file: 
        pickle.dump(og_targets, file)
        
def indx_incorrect(C,O,R):

    val_ind_1 = (O != C).nonzero(as_tuple=True)[0]
    val_ind_2 = (O != R).nonzero(as_tuple=True)[0]

    
    return val_ind_1,val_ind_2

# In[6]:


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Select model_type, datasets and corruption.")

    corrution_prob =[0.0,0.2,0.4,0.6,0.8,1.0]
    model_type = ['CNN','AlexNet']
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
        
        
    t0 = time.time()
    torch.manual_seed(42)

    n=0.99
    runs=4
    early_stopping=True
    num_epoch=30 
    
    
    corrupted_train,test_loader,og_targets,retrain_path,network_path,batch_size=originial_data(
        type_network,ds)

    os.makedirs(retrain_path,exist_ok=True)

    results_folder=f'angle_results_retrain_30'
    os.makedirs(results_folder,exist_ok=True)
    os.makedirs(f'{results_folder}/{ds}_{type_network}',exist_ok=True)

    #results foldersfor corrupted subspace
    results_corr={}
    os.makedirs(f'{results_folder}/{ds}_{type_network}/results_{corrupt}',exist_ok=True)
    os.makedirs(f'{results_folder}/{ds}_{type_network}/results_{corrupt}/angle_results',exist_ok=True)
    results_corr['angle_1']=f'{results_folder}/{ds}_{type_network}/results_{corrupt}/angle_results'
    
    results_corr['pencent']=f'{results_folder}/{ds}_{type_network}/results_{corrupt}/percentages_data_change'
    os.makedirs(results_corr['pencent'],exist_ok=True)

    #results folders for original subspace
    results_org={}
    os.makedirs(f'{results_folder}/{ds}_{type_network}/results_{corrupt}/angle_results_exp3',exist_ok=True)
    results_org['angle_1']=f'{results_folder}/{ds}_{type_network}/results_{corrupt}/angle_results_exp3'
    
    
    

    test_loader_80,test_loader_20=test_split(test_loader)

    
#     for run in range(1,runs): 
    #path for temprary activation storage
    temp_path = f'Network_data/retrain_{corrupt}_{ds}_{type_network}_{run}'
    os.makedirs(temp_path,exist_ok=True)
#     print(run)
    epoch=epoch_number(network_path,corrupt,run)
    dummy_model=model_build(type_network,ds)
    dummy_model.load_state_dict(torch.load(
        f'{network_path}/{corrupt}/Run_{run}/model_{epoch}.pth',map_location=dev))

    #results angle on 80
    cnn_create.loading_saving_activations(temp_path,
                                          dummy_model,corrupted_train,
                                          test_loader_80,og_targets,
                                          dev,type_network)

    angle.angle_corrupt_test_only(type_network,temp_path,
                             run,results_corr,n,epoch,ds,80)   
    angle.angle_original_test_only(type_network,temp_path,
                              run,results_org,n,epoch,ds,80)


    #results angle on 20
    cnn_create.loading_saving_activations(temp_path,
                                          dummy_model,corrupted_train,
                                          test_loader_20,og_targets,
                                          dev,type_network)

    angle.angle_corrupt_test_only(type_network,temp_path,
                             run,results_corr,n,epoch,ds)   
    angle.angle_original_test_only(type_network,temp_path,
                              run,results_org,n,epoch,ds)

    del dummy_model

    print("angle testing done")
    #identifing layer highest index on testing data
    layer_highest_corr,layer_highest_org=select_layer_idx(type_network,ds,
                                                     results_corr,
                                                          results_org,epoch)



    #for corrupted subspace
    name_type='corrupt'
    y_train_corr=angle.angle_relabel(type_network,temp_path,run,
                  n,epoch,name_type,layer_highest_corr,ds)

    subset_loader=relabels_dataset(corrupted_train,y_train_corr,batch_size)

    print("relabeling of training dataset done corrupted")
    run_path=f'{retrain_path}/{corrupt}/corr_subspace/Run_{run}'
    accuracy_path=f'{results_folder}/{ds}_{type_network}/results_{corrupt}/Retrain_Acc_corr'
    os.makedirs(run_path,exist_ok=True)
    os.makedirs(accuracy_path,exist_ok=True)

    model=model_build(type_network,ds)
    model.load_state_dict(torch.load(
        f'{network_path}/{corrupt}/Run_{run}/model_{epoch}.pth',
        map_location =dev))

    loss_func,optimizer=model_params(type_network,ds)

    training_process(subset_loader,test_loader_80,test_loader_20,
                     model,loss_func,optimizer,
                     run_path,accuracy_path,run,dev,
                     epoches=num_epoch,early_stopping=early_stopping)
    print("corrupted retraining process done")
    #for original subspace
    name_type='original'
    y_train_org=angle.angle_relabel(type_network,temp_path,run,
                  n,epoch,name_type,layer_highest_org,ds)


    subset_loader=relabels_dataset(corrupted_train,y_train_org,batch_size)
    print("relabeling of training dataset done original")


    run_path=f'{retrain_path}/{corrupt}/org_subspace/Run_{run}'
    os.makedirs(run_path,exist_ok=True)
    accuracy_path=f'{results_folder}/{ds}_{type_network}/results_{corrupt}/Retrain_Acc_org'
    os.makedirs(accuracy_path,exist_ok=True)

    model=model_build(type_network,ds)
    model.load_state_dict(torch.load(
        f'{network_path}/{corrupt}/Run_{run}/model_{epoch}.pth',
        map_location =dev))


    loss_func,optimizer=model_params(type_network,ds)

    # model testing accuracy (saved in org values)
    test_acc(test_loader_80,model,loss_func,accuracy_path,run,dev)

    training_process(subset_loader,test_loader_80,test_loader_20,
                     model,loss_func,optimizer,run_path,
                     accuracy_path,run,dev,
                     epoches=num_epoch,early_stopping=early_stopping)

    if corrupt!=0.0:
        results_path=results_corr['pencent']
        percentage_captured(temp_path,y_train_corr,y_train_org,
                            run,results_path)

    print("original retraining process done")

    shutil.rmtree(temp_path)
    t1=time.time()
    print("time taken for run : ",t1-t0)
    print(f"corrupt {corrupt}, run {run} done")




