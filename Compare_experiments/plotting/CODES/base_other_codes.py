import os

def path_network_fn(type_network,ds):
    if type_network=='MLP':
        if ds=='MNIST':
            network_path = '/mnt/NVME2TB/simran/PCA/MLP_MNIST/Network'

        if ds=='CIFAR10':
            network_path = '/mnt/NVME2TB/simran/PCA/MLP_CIFAR10/Network'
            
    if type_network =='CNN':
        if ds=='FashionMNIST':
            network_path = '/mnt/NVME2TB/simran/PCA/FashionMNIST/Network'
            
        if ds=='MNIST':
            network_path = '/mnt/NVME2TB/simran/PCA/MNIST/Network'
            
        if ds=='CIFAR10':
            network_path = '/mnt/8TB/simran/PCA/CIFAR_10_Wodrop/Network'
            
    if type_network=='AlexNet':
        if ds=='TinyImageNet':
            network_path = '/mnt/NVME2TB/simran/PCA/TinyImagenet_Alexnet/Network'
        if ds=='CIFAR100':
            network_path = '/mnt/8TB/simran/PCA/CIFAR_100/Network'
        
    return network_path

def trained_model(network_path,corrupt,run):
    model_list=os.listdir(f'{network_path}/{corrupt}/Run_{run}')
    initialized_model = 'initialized_model.pth'
    remaining_models = [model for model in model_list if model != initialized_model]
    # Sort remaining models numerically
    remaining_models.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    return remaining_models[-1]


def epochnumber(epoch):
    if epoch!='initialized_model.pth':
        return int(epoch.split('_')[1].split('.')[0])+1
    else:
        return 0
def get_epoch_present(type_network,ds,corrupt,run):
    network_path=path_network_fn(type_network,ds)
    epoch=trained_model(network_path,corrupt,run)
    epoch_present=epochnumber(epoch)
    return epoch_present


def layer_name(type_network,ds):
    if type_network =='AlexNet':
        
        data_layer=['after_flatten','after_relu_fc1','after_relu_fc2'] #'input_layer',
        pca_layer=['flattern','fc1','fc2']#'input',
        if ds=='TinyImagenet':
            num_class=200
        else:
            num_class=100
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