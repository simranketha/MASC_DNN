import pickle
import numpy as np
import os

import torch

import pickle
def results_fol_name(results_folder,corrupt):
    
    #results foldersfor corrupted subspace
    results_corr={}
    base_code=f'{results_folder}/results_{corrupt}'
    
    results_corr['angle_1']=f'{base_code}/angle_results'
    os.makedirs(results_corr['angle_1'],exist_ok=True)
    
    #original training labels + corrupted training subspaces : exp2
    results_corr['angle_2']=f'{base_code}/angle_results_exp2'
    os.makedirs(results_corr['angle_2'],exist_ok=True)
    #results folders for original subspace
    results_org={}
    results_org['angle_1']=f'{base_code}/angle_results_exp3'
    results_org['angle_2']=f'{base_code}/angle_results_exp3'
    os.makedirs(results_org['angle_2'],exist_ok=True)
    return results_corr,results_org

def trained_model(network_path,corrupt,run):
    model_list=os.listdir(f'{network_path}/{corrupt}/Run_{run}')
    initialized_model = 'initialized_model.pth'
    remaining_models = [model for model in model_list if model != initialized_model]
    # Sort remaining models numerically
    remaining_models.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    return remaining_models[-1]


def layer_name(type_network):
    if type_network =='AlexNet':
        data_layer=['after_flatten','after_relu_fc1','after_relu_fc2'] #'input_layer',
        pca_layer=['flattern','fc1','fc2']#'input',
        num_class=200
    #cnn model
    if type_network =='CNN':
        data_layer=['input_fc_0','output_fc_0_after_noise_relu',
                    'output_fc_1_after_noise_relu','output_fc_2_after_noise_relu']
        #'input_layer',
        pca_layer=['flattern','fc1','fc2','fc3']#'input',
        num_class=10
    #mlp model
    if type_network =='MLP':
        data_layer=['after_relu_fc1', 
                    'after_relu_fc2','after_relu_fc3', 
                    'after_relu_fc4'] #'input_layer',
        pca_layer=['fc1','fc2','fc3','fc4']#'input',
        num_class=10
        
    return pca_layer,data_layer,num_class




def data_loading_test(temp_path, d_layer):
    with open(f'{temp_path}/{d_layer}_test.pkl', 'rb') as file: 
        myvar = pickle.load(file)

    layer_output_test = [torch.tensor(np.array(myvar[values]), dtype=torch.float32) for values in range(len(myvar))]
    layer_output_test = torch.stack(layer_output_test).cuda()

    with open(f'{temp_path}/y_value_corrupted_test.pkl', 'rb') as file: 
        myvar = pickle.load(file)

    original_test_labels = [torch.tensor(np.array(myvar[values]), dtype=torch.long) for values in range(len(myvar))]
    original_test_labels = torch.stack(original_test_labels).squeeze().cuda()
    
    return layer_output_test, original_test_labels

def data_loading_train(temp_path, d_layer):
    # training data
    with open(f'{temp_path}/{d_layer}_train.pkl', 'rb') as file: 
        myvar = pickle.load(file)

    # Stack all layer outputs -> shape [N, feature_dim]
    layer_output = [torch.tensor(np.array(myvar[values]), dtype=torch.float32) for values in range(len(myvar))]
    layer_output = torch.stack(layer_output).cuda()

    # Original labels
    with open(f'{temp_path}/y_value_original_train.pkl', 'rb') as file: 
        myvar = pickle.load(file)

    og_targets = [torch.tensor(np.array(myvar[values]), dtype=torch.long) for values in range(len(myvar))]
    og_targets = torch.stack(og_targets).squeeze().cuda()

    # Corrupted labels
    with open(f'{temp_path}/y_value_corrupted_train.pkl', 'rb') as file: 
        myvar = pickle.load(file)

    cor_targets = [torch.tensor(np.array(myvar[values]), dtype=torch.long) for values in range(len(myvar))]
    cor_targets = torch.stack(cor_targets).squeeze().cuda()

    return layer_output, og_targets, cor_targets