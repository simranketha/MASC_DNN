import numpy as np
import copy
import torch
from torchvision.datasets import CIFAR10,FashionMNIST,MNIST,CIFAR100,ImageNet
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import pickle

import os
import torchvision.datasets as datasets

# ########### models ##################


class NgnCnn(nn.Module):
    def __init__(
        self,
        layer_size=250,
        channels=3,
        control=False,
        seed=0,
        excite=False,
        neural_noise=None,
        dropout=0,
    ):
        torch.manual_seed(seed)
        super(NgnCnn, self).__init__()
        # parameters
        self.ablate = False
        self.dropout = dropout
        self.channels = channels
        self.excite = excite
        self.n_new = 0
        self.control = False
        if self.control:
            self.idx_control = np.random.choice(
                range(layer_size), size=8, replace=False
            )
        self.neural_noise = neural_noise

        # 3@16x16
        self.conv1 = nn.Conv2d(channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.AvgPool2d(kernel_size=1, stride=1)

        self.layer_size = layer_size

        self.fc_new_in = nn.ModuleList()
        self.fc_new_out = nn.ModuleList()

        if self.channels == 3:
            self.cnn_output = 64 * 4 * 4
        elif self.channels == 1:
            self.cnn_output = 64 * 9
        # three fully connected layers
        self.fcs = nn.ModuleList(
            [
                nn.Linear(self.cnn_output, self.layer_size),  # 0
                nn.Linear(self.layer_size, self.layer_size),  # 1 on dim 2 neurogenesis
                nn.Linear(self.layer_size, self.layer_size),  # 2
            ]
        )
        self.fc3 = nn.Linear(self.layer_size, 10, bias=False)

        self.out_conv1 = None
        self.out_conv2 = None
        self.out_conv3 = None
        self.out_conv4 = None
        self.out_conv5 = None
        self.out_conv6 = None
        self.out_pool1 = None
        self.out_pool2 = None
        self.out_pool3 = None
        self.out_pool4 = None

        self.out_fcs = dict()

        self.out_before_fc3 = None


    def get_intermediate_states(self):
        return [self.input,
                self.out_conv1 , 
              self.out_conv2 ,
              self.out_conv3,
              self.out_conv4 ,
              self.out_conv5,
              self.out_conv6 ,
              self.out_pool1 ,
              self.out_pool2 ,
              self.out_pool3 ,
              self.out_pool4,
              self.out_fcs,
              self.out_before_fc3 ]
    
    def forward(self, x, extract_layer=None):
        self.input = x.clone().detach()
        x = F.relu(self.conv1(x))
        self.out_conv1 = x.clone().detach()
        
        x = F.relu(self.conv2(x))
        self.out_conv2 = x.clone().detach()
        
        x = self.pool(x)
        self.out_pool1 = x.clone().detach()
        
        x = F.relu(self.conv3(x))
        self.out_conv3 = x.clone().detach()
        
        x = F.relu(self.conv4(x))
        self.out_conv4 = x.clone().detach()
        
        x = self.pool2(x)
        self.out_pool2 = x.clone().detach()
        
        x = F.relu(self.conv5(x))
        self.out_conv5 = x.clone().detach()

        x = F.relu(self.conv6(x))
        self.out_conv6 = x.clone().detach()
        x = self.pool3(x)
        self.out_pool3 = x.clone().detach()
        
        x = self.pool4(x)
        self.out_pool4 = x.clone().detach()
        

        x = x.view(-1, self.cnn_output)
        self.out_flatten = x.clone().detach()
        

        for ix, fc in enumerate(self.fcs):
            self.out_fcs[f'input_fc_{ix}'] = x.clone().detach()
            x = fc(x)
            self.out_fcs[f'output_fc_{ix}'] = x.clone().detach()
            
            if self.neural_noise is not None and ix == 0 and self.training:
                mean, std = self.neural_noise
                noise = torch.zeros_like(x, device=dev)
                noise = noise.log_normal_(mean=mean, std=std)
                x = x * noise
            self.out_fcs[f'output_fc_{ix}_after_noise'] = x.clone().detach()
            
            x = F.relu(x)
            self.out_fcs[f'output_fc_{ix}_after_noise_relu'] = x.clone().detach()

            if self.excite and ix == 1 and self.n_new and self.training:
                idx = self.idx_control if self.control else self.idx
                excite_mask = torch.ones_like(x)
                excite_mask[:, idx] = self.excite
                excite_mask.to(dev)
                x = x * excite_mask
            self.out_fcs[f'output_fc_{ix}_after_excite'] = x.clone().detach()

            if self.dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = torch.renorm(x, 1, 1, 3)  # max norm
            self.out_fcs[f'output_fc_{ix}_after_dropout'] = x.clone().detach()

            # for ablation experiments
            if self.ablate:
                if ix == 1:
                    activation_size = x.size()[1]
                    if self.ablation_mode == "random":
                        ablate_size = int(self.ablation_prop * activation_size)
                        indices = np.random.choice(
                            range(activation_size),
                            size=size,
                            replace=False,
                        )
                    if self.ablation_mode == "targetted":
                        indices = self.ablate_indices
                    x[:, indices] = 0
            if extract_layer == ix:
                return x
        self.out_before_fc3 = x.clone().detach()
        x = self.fc3(x)

        return x

class AlexNet(nn.Module):

  def __init__(self, num_classes=100,tiny_imagenet=False,dropout=False):
    super(AlexNet, self).__init__()
    if tiny_imagenet:
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
#             nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
#             nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    else:
        self.features = nn.Sequential(
          nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
          nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
          nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
        )
    self.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(256 * 1 * 1, 4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(inplace=True),
      nn.Linear(4096, num_classes),
    )
    if dropout:
        self.drop=nn.Dropout(0.5)
    self.fc1 = nn.Linear(256 * 1 * 1, 4096)
    self.fc2 = nn.Linear(4096, 4096)
    self.fc3 = nn.Linear(4096, num_classes)
    
    
    self.input_img = None
    self.after_flatten = None
    self.before_relu_fc1 = None
    self.after_relu_fc1 = None
    self.before_relu_fc2 = None
    self.after_relu_fc2 = None
    self.before_relu_fc3 = None
    self.after_relu_fc3 = None
    if dropout:
        self.after_dropout_flatten=None
        self.after_dropout_fc1=None
        self.after_dropout_fc2=None

  def forward(self, x,dropout):
    if dropout:
        self.input_img = x.clone().detach()
        x = self.features(x)
        x = torch.flatten(x, 1)

        self.after_flatten = x.clone().detach()
        x = self.drop(x)
        self.after_dropout_flatten = x.clone().detach()

        x = self.fc1(x)
        self.before_relu_fc1 = x.clone().detach()
        x = F.relu(x)
        self.after_relu_fc1 = x.clone().detach()
        x = self.drop(x)
        self.after_dropout_fc1 = x.clone().detach()

        x = self.fc2(x)
        self.before_relu_fc2 = x.clone().detach()
        x = F.relu(x)
        self.after_relu_fc2 = x.clone().detach()
        x = self.drop(x)
        self.after_dropout_fc2 = x.clone().detach()

        x = self.fc3(x)
        self.before_relu_fc3 = x.clone().detach()

    else:
        
        self.input_img = x.clone().cpu().detach()
        x = self.features(x)
        x = torch.flatten(x, 1)

        self.after_flatten = x.clone().cpu().detach()
        x = self.fc1(x)
        self.before_relu_fc1 = x.clone().cpu().detach()
        x = F.relu(x)
        self.after_relu_fc1 = x.clone().cpu().detach()

        x = self.fc2(x)
        self.before_relu_fc2 = x.clone().cpu().detach()
        x = F.relu(x)
        self.after_relu_fc2 = x.clone().cpu().detach()

        x = self.fc3(x)
        self.before_relu_fc3 = x.clone().cpu().detach()

    
    return x

  def get_intermediate_states(self,dropout):
    if dropout:
        return [
        self.input_img,
        self.after_flatten,
        self.after_dropout_flatten,
        self.after_relu_fc1,
        self.after_dropout_fc1,
        self.after_relu_fc2,
        self.after_dropout_fc2,
        self.before_relu_fc3]
    else:
        return [
        self.input_img,
        self.after_flatten,
        self.after_relu_fc1,
        self.after_relu_fc2,
        self.before_relu_fc3]



def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["alexnet"]))
#     
    #wodrop
    

    model.classifier[0] = nn.Linear(256 * 1 * 1, 4096)
    model.classifier[4] = nn.Linear(4096, 200)
    #wdrop
#     model.classifier[1] = nn.Linear(256 * 1 * 1, 4096)
#     model.classifier[6] = nn.Linear(4096, 200)

    return model





# ########### datasets ##################

# In[11]:

"""**Load dataset**"""
def load_data(
    mode,
    data_folder="./data",
    num_workers=16,
    batch_size=50,
    split=0.1,
    seed=23,
    fashion=False,
    cifar10=False,
    mnist=False,
    imagenet=False
):
    """
    Helper function to read in image dataset, and split into
    training, validation and test sets.
    ===
    mode: str, ['validation', 'test]. If 'validation', training data
         will be divided based on split parameter.
         If test, .valid = None, and all training data is used for training
    split: float, where 0 < split < 1. Where train = split * num_samples
        and valid = (1 - split) * num_samples
    seed: int, random seed to generate validation/training split
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    assert mode in ["validation", "test"], "mode not validation nor test"

    if fashion:
        trainset = torchvision.datasets.FashionMNIST(
            data_folder,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            ),
        )
        testset = torchvision.datasets.FashionMNIST(
            data_folder,
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            ),
        )
        print("Loaded FMNIST dataset")
    elif cifar10:
        trainset = torchvision.datasets.CIFAR10(
            root=data_folder, train=True, download=True, transform=transform
        )
        

        testset = torchvision.datasets.CIFAR10(
            root=data_folder, train=False, download=True, transform=transform
        )
    elif mnist:
        transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.MNIST(
            root=data_folder, train=True, download=True, transform=transform
        )

        testset = torchvision.datasets.MNIST(
            root=data_folder, train=False, download=True, transform=transform
        )
    elif imagenet:
        data_folder="/mnt/2TB/DeepLearningExamples/PyTorch/Classification"
        # Initialize transformations for data augmentation
#         transform = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406],
#                                  [0.229, 0.224, 0.225])
#         ])
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        trainset = torchvision.datasets.ImageNet(
            root=data_folder, train=True, download=False, transform=transform
        )

        testset = torchvision.datasets.ImageNet(
            root=data_folder, train=False, download=False, transform=transform
        )
        
    else:
        trainset = torchvision.datasets.CIFAR100(
            root=data_folder, train=True, download=True, transform=transform
        )
        

        testset = torchvision.datasets.CIFAR100(
            root=data_folder, train=False, download=True, transform=transform
        )
    print('train len: ',len(trainset))
    print('test len: ',len(testset))

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True, 
        pin_memory=True,
        
    )

    if mode == "validation":
        from sklearn.model_selection import train_test_split

        num_train = 50000
        indices = list(range(num_train))

        train_idx, valid_idx = train_test_split(
            indices, test_size=split, random_state=seed
        )

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            sampler=train_sampler,
            pin_memory=True,
        )

        validloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=valid_sampler,
            drop_last=True, 
            pin_memory=True,
        )
        print("Created data loaders")
        return trainloader, validloader, testloader

    elif mode == "test":
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True, pin_memory=True,
        )
        print("Created data loaders")
        return trainloader, testloader

    
    
class Imagenet_data(object):
    def __init__(
        self,
        mode="validation",
        data_folder="/mnt/2TB/DeepLearningExamples/PyTorch/Classification",
        batch_size=50,
        imagenet=False,
        num_workers=16,
        split=0.1,
        seed=23,
    ):
        if mode == "validation":
            self.train, self.valid, self.test = load_data(
                mode=mode,
                data_folder=data_folder,
                batch_size=batch_size,
                num_workers=num_workers,
                split=split,
                fashion=fashion,
                seed=seed,
            )
        elif mode == "test":
            self.train, self.test = load_data(
                mode=mode,
                data_folder=data_folder,
                seed=seed,
                imagenet=imagenet,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            self.valid = None
            
class ImagenetCorrupted(object):
    def __init__(self, corrupt_prob, num_classes=1000, **kwargs):
        super().__init__(**kwargs)
        self.original_targets = []
        self.updated_targets = []
        self.corrupt_prob = corrupt_prob
        if corrupt_prob >= 0:
            self.n_classes = num_classes
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        self.original_targets = copy.deepcopy(labels)
        np.random.seed(42)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        labels = [int(x) for x in labels]
        self.targets = labels
        self.updated_targets = copy.deepcopy(np.array(labels))

    def get_targets(self):
        return self.corrupt_prob, self.original_targets, self.updated_targets



class ImagenetCorrupted(ImageNet):
    def __init__(self, root="/mnt/2TB/DeepLearningExamples/PyTorch/Classification", split='train', 
                 corrupt_prob=0.0, transform=None):

        self.root = os.path.join(root, split)
        self.dataset = datasets.ImageFolder(self.root, transform=transform)
        self.targets = np.array([label for _, label in self.dataset.samples])
        self.corrupt_prob = corrupt_prob
        self.n_classes = len(self.dataset.classes)

        # Store original and corrupted targets
        self.original_targets = copy.deepcopy(self.targets)
        self.corrupted_targets = copy.deepcopy(self.targets)

        # Corrupt the labels if corrupt_prob > 0
        if corrupt_prob > 0:
            self._corrupt_labels()

    def _corrupt_labels(self):
        """
        Corrupt labels with a probability defined by corrupt_prob.
        """
        np.random.seed(42)  # Ensure reproducibility
        mask = np.random.rand(len(self.targets)) < self.corrupt_prob
        corrupt_labels = np.random.choice(self.n_classes, size=mask.sum(), replace=True)
        self.corrupted_targets[mask] = corrupt_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the possibly corrupted label.
        """
        path, _ = self.dataset.samples[index]
        image = self.dataset.loader(path)
        if self.dataset.transform is not None:
            image = self.dataset.transform(image)
        label = self.corrupted_targets[index]
        return image, label

    def __len__(self):
        return len(self.dataset.samples)

    def get_targets(self):
        """
        Returns:
            tuple: (original_targets, corrupted_targets)
        """
        return self.corrupt_prob, self.original_targets, self.corrupted_targets



    
class CIFAR100_data(object):
    def __init__(
        self,
        mode="validation",
        data_folder="./data",
        batch_size=50,
        fashion=False,
        num_workers=16,
        split=0.1,
        seed=23,
        cifar10=False,
        
    ):
        if mode == "validation":
            self.train, self.valid, self.test = load_data(
                mode=mode,
                data_folder=data_folder,
                batch_size=batch_size,
                num_workers=num_workers,
                split=split,
                fashion=fashion,
                seed=seed,
                cifar10=cifar10,
                
            )
        elif mode == "test":
            self.train, self.test = load_data(
                mode=mode,
                data_folder=data_folder,
                seed=seed,
                fashion=fashion,
                batch_size=batch_size,
                num_workers=num_workers,
                cifar10=cifar10,
            )
            self.valid = None



class Cifar10_data(object):
    def __init__(
        self,
        mode="validation",
        data_folder="./data",
        batch_size=50,
        fashion=False,
        num_workers=16,
        split=0.1,
        seed=23,
    ):
        if mode == "validation":
            self.train, self.valid, self.test = load_data(
                mode=mode,
                data_folder=data_folder,
                batch_size=batch_size,
                num_workers=num_workers,
                split=split,
                fashion=fashion,
                seed=seed,
            )
        elif mode == "test":
            self.train, self.test = load_data(
                mode=mode,
                data_folder=data_folder,
                seed=seed,
                fashion=fashion,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            self.valid = None

class FashionMNISTCorrupted(FashionMNIST):
    def __init__(self, corrupt_prob, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.original_targets = []
        self.updated_targets = []
        self.corrupt_prob = corrupt_prob
        if corrupt_prob >= 0:
            self.n_classes = num_classes
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        self.original_targets = copy.deepcopy(labels)
        np.random.seed(42)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        labels = [int(x) for x in labels]
        self.targets = labels
        self.updated_targets = copy.deepcopy(np.array(labels))

    def get_targets(self):
        return self.corrupt_prob, self.original_targets, self.updated_targets

class MNISTCorrupted(MNIST):
    def __init__(self, corrupt_prob, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.original_targets = []
        self.updated_targets = []
        self.corrupt_prob = corrupt_prob
        if corrupt_prob >= 0:
            self.n_classes = num_classes
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        self.original_targets = copy.deepcopy(labels)
        np.random.seed(42)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        labels = [int(x) for x in labels]
        self.targets = labels
        self.updated_targets = copy.deepcopy(np.array(labels))

    def get_targets(self):
        return self.corrupt_prob, self.original_targets, self.updated_targets

    
class CIFAR10Corrupted(CIFAR10):
    def __init__(self, corrupt_prob, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.original_targets = []
        self.updated_targets = []
        self.corrupt_prob = corrupt_prob
        if corrupt_prob >= 0:
            self.n_classes = num_classes
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        self.original_targets = copy.deepcopy(labels)
        np.random.seed(42)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        labels = [int(x) for x in labels]
        self.targets = labels
        self.updated_targets = copy.deepcopy(np.array(labels))

    def get_targets(self):
        return self.corrupt_prob, self.original_targets, self.updated_targets


class TinyImagenetCorrupted(Dataset):
    def __init__(self, root="tiny-imagenet-200", split='train', corrupt_prob=0.0, transform=None):
        """
        Args:
            root (str): Root directory of the Tiny-ImageNet dataset.
            split (str): Dataset split, one of ['train', 'val', 'test'].
            corrupt_prob (float): Probability of corrupting a label.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = os.path.join(root, split)
        self.dataset = datasets.ImageFolder(self.root, transform=transform)
        self.targets = np.array([label for _, label in self.dataset.samples])
        self.corrupt_prob = corrupt_prob
        self.n_classes = len(self.dataset.classes)

        # Store original and corrupted targets
        self.original_targets = copy.deepcopy(self.targets)
        self.corrupted_targets = copy.deepcopy(self.targets)

        # Corrupt the labels if corrupt_prob > 0
        if corrupt_prob > 0:
            self._corrupt_labels()

    def _corrupt_labels(self):
        """
        Corrupt labels with a probability defined by corrupt_prob.
        """
        np.random.seed(42)  # Ensure reproducibility
        mask = np.random.rand(len(self.targets)) < self.corrupt_prob
        corrupt_labels = np.random.choice(self.n_classes, size=mask.sum(), replace=True)
        self.corrupted_targets[mask] = corrupt_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the possibly corrupted label.
        """
        path, _ = self.dataset.samples[index]
        image = self.dataset.loader(path)
        if self.dataset.transform is not None:
            image = self.dataset.transform(image)
        label = self.corrupted_targets[index]
        return image, label

    def __len__(self):
        return len(self.dataset.samples)

    def get_targets(self):
        """
        Returns:
            tuple: (original_targets, corrupted_targets)
        """
        return self.corrupt_prob, self.original_targets, self.corrupted_targets


class CIFAR100Corrupted(CIFAR100):
    def __init__(self, corrupt_prob, num_classes=100, **kwargs):
        super().__init__(**kwargs)
        self.original_targets = []
        self.updated_targets = []
        self.corrupt_prob = corrupt_prob
        if corrupt_prob >= 0:
            self.n_classes = num_classes
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        self.original_targets = copy.deepcopy(labels)
        np.random.seed(42)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        labels = [int(x) for x in labels]
        self.targets = labels
        self.updated_targets = copy.deepcopy(np.array(labels))

    def get_targets(self):
        return self.corrupt_prob, self.original_targets, self.updated_targets





def get_cifar_dataloaders_corrupted(corrupt_prob=0, batch_size=50,fashion=False,cifar10=False,
                                    mnist=False,tiny_imagenet=False,
                                    imagenet=False):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data_foldername='data'
    data_foldername_tiny='data/tiny-imagenet-200'
    data_foldername_imagenet='data/imagenet'
    if fashion:
        transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
        trainset = FashionMNISTCorrupted(root=data_foldername, train=True, download=True, transform=transform,
                                    corrupt_prob=corrupt_prob)
        
        testset = FashionMNISTCorrupted(root=data_foldername, train=False, download=True, transform=transform,
                                   corrupt_prob=corrupt_prob)  
    elif cifar10:
        trainset = CIFAR10Corrupted(root=data_foldername, train=True, download=True, transform=transform,
                                    corrupt_prob=corrupt_prob)
        
        testset = CIFAR10Corrupted(root=data_foldername, train=False, download=True, transform=transform,
                                   corrupt_prob=corrupt_prob)
    elif tiny_imagenet:
        # Use TinyImagenetCorrupted class for Tiny-ImageNet with the given corruption probability
        transform = transforms.Compose(
            [transforms.ToTensor()]
        )
        trainset = TinyImagenetCorrupted(root=data_foldername_tiny, split='train', corrupt_prob=corrupt_prob, transform=transform)
        testset = TinyImagenetCorrupted(root=data_foldername_tiny, split='val', corrupt_prob=0.0, transform=transform)  # Validation set as test
    
    elif imagenet:
        # Use Imagenet Corrupted class for ImageNet with the given corruption probability
        # Initialize transformations for data augmentation
#         transform = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406],
#                                  [0.229, 0.224, 0.225])
#         ])
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainset = ImagenetCorrupted(root=data_foldername_imagenet, split='train', transform=transform,
                                    corrupt_prob=corrupt_prob)
        testset = ImagenetCorrupted(root=data_foldername_imagenet, split='val', transform=transform,
                                   corrupt_prob=corrupt_prob)
        
    elif mnist:
        transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
        trainset = MNISTCorrupted(root=data_foldername, train=True, download=True, transform=transform,
                                    corrupt_prob=corrupt_prob)
        testset = MNISTCorrupted(root=data_foldername, train=False, download=True, transform=transform,
                                   corrupt_prob=corrupt_prob)
    else:
        trainset = CIFAR100Corrupted(root=data_foldername, train=True, download=True, transform=transform,
                                    corrupt_prob=corrupt_prob)
        testset = CIFAR100Corrupted(root=data_foldername, train=False, download=True, transform=transform,
                                   corrupt_prob=corrupt_prob)
    og_prob, og_targets, cor_targets = trainset.get_targets()
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader, og_prob, og_targets, cor_targets
-----------------------------------------------------

#mdoel and dataset (training and layer out)

def model_build(type_network,ds=None,dropout):
    if dropout:
        dropvalue=0
    else:
        dropvalue=0.2
    if type_network=='CNN':
        if ds=='CIFAR10':
            dummy_model = cnn_create.NgnCnn(dropout=dropvalue)
        else:
            dummy_model = cnn_create.NgnCnn(channels=1,dropout=dropvalue)
        optimizer = torch.optim.Adam(model.parameters(),lr=0.0002)
    loss_func = nn.CrossEntropyLoss()
    if type_network=='AlexNet':
        if ds=='TinyImageNet':
            dummy_model = cnn_create.AlexNet(num_classes=200, tiny_imagenet=True)
            optimizer=optim.Adam(model.parameters(),betas=(0.9, 0.999),lr = 0.0001)
            loss_func=nn.CrossEntropyLoss().to(device)

        if ds=='CIFAR100':
            dummy_model = cnn_create.AlexNet(num_classes=100,dropout=dropout)
            optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
            loss_func = nn.CrossEntropyLoss()
    return dummy_model,loss_func,optimizer

def path_model(ds,dropout):
    if ds=='MNIST':
        path ='/mnt/SSD2TB/simran/PCA/MNIST_CNN'
        
        if dropout:
            path='/mnt/SSD2TB/simran/PCA/MNIST_CNN_dropout'
            
    if ds=='FashionMNIST':
        path = '/mnt/8TB/simran/PCA/FashionMNIST' #check this
        if dropout:
            path='/mnt/8TB/simran/PCA/FashionMNIST_d_0_2'
            
    if ds=='CIFAR10':
        path = '/mnt/8TB/simran/PCA/CIFAR_10_Wodrop'
        if dropout:
            path = '/mnt/8TB/simran/PCA/CIFAR_10_CNN'
    if ds=='CIFAR100':
        path = '/mnt/8TB/simran/PCA/CIFAR_100' 
        
        if dropout:
            path = '/mnt/SSD2TB/simran/PCA/CIFAR_100_dropout'
            
    if ds=='TinyImageNet':
        path = '/mnt/SSD2TB/simran/PCA/TinyImagenet_Alexnet'

    os.makedirs(path,exist_ok=True)
    network_path = os.path.join(path,'Network')
    os.makedirs(network_path, exist_ok=True)
    res_path = os.path.join(path,'Accuracy_results')
    
    return path,network_path,res_path

def test_loading(ds):
    if ds=='MNIST':
        batch_size = 128
        _, test_loader = load_data("test",num_workers=1,batch_size=4,mnist=True)
        
    if ds=='FashionMNIST':
        batch_size = 128
        _, test_loader = load_data("test",num_workers=1,batch_size=4,fashion=True)
        
    if ds=='CIFAR10':
        batch_size = 32
        _, test_loader = load_data("test",num_workers=1,batch_size=4,cifar10=True)
    if ds=='CIFAR100':
        batch_size =128
        _, test_loader = load_data("test",num_workers=1,batch_size=4)
        
    if ds=='TinyImageNet':
        batch_size = 500
        _, test_loader , _, _, _ = get_cifar_dataloaders_corrupted(batch_size=batch_size,
                                                                   tiny_imagenet=True)
    
    return test_loader,batch_size


def train_loading(ds,batch_size,corrupt):
    if ds=='MNIST':
        corrupted_train, _ ,_,og_targets,cor_targets = get_cifar_dataloaders_corrupted(corrupt,
                                                                                            batch_size=batch_size,mnist=True)
    if ds=='FashionMNIST':
        corrupted_train, _ , _, og_targets, cor_targets = get_cifar_dataloaders_corrupted(corrupt, batch_size=batch_size,fashion=True)
        
    if ds=='CIFAR10':
        corrupted_train, _ , _, og_targets, cor_targets = get_cifar_dataloaders_corrupted(corrupt, batch_size=batch_size,cifar10=True)
    
    if ds=='TinyImageNet':
        corrupted_train, _ , _, og_targets, cor_targets = get_cifar_dataloaders_corrupted(corrupt, batch_size=batch_size,tiny_imagenet=True)
    if ds=='CIFAR100':
        corrupted_train, _ , _, og_targets, cor_targets = get_cifar_dataloaders_corrupted(corrupt, batch_size=batch_size)
    return corrupted_train, og_targets, cor_targets

---------------------------------------

def data_saving_drop(path,loader,dummy_model,name,dev,type_network):
    if type_network=='AlexNet':
        dummy_output =None
        output0=[]
        output1=[]
        output2=[]
        output3=[]
        output4=[]
        output5=[]
        output6=[]
        y_value=[]
        dummy_model.to(dev)
        for x,y in loader:
            output0.append(np.array(x).reshape(-1))
            dummy_output = dummy_model(x.to(dev))
            out_fcs = dummy_model.get_intermediate_states()

            output1.append(out_fcs[1][0])
            output2.append(out_fcs[2][0])
            output3.append(out_fcs[3][0])
            output4.append(out_fcs[4][0])
            output5.append(out_fcs[5][0])
            output6.append(out_fcs[6][0])
            y_value.append(y[0])

        data_layer_name=['input_layer',
                         'after_flatten','after_dropout_flatten',
                         'after_relu_fc1', 'after_dropout_fc1',
                         'after_relu_fc2','after_dropout_fc2',
                         'y_value_corrupted']
        
        outputs = [output0, output1, output2, output3, output4,output5,output6, y_value]
        
    if type_network=='CNN':
        dummy_output =None
        output0=[]
        output1=[]
        output2=[]
        output3=[]
        output4=[]
        output5=[]
        output6=[]
        output7=[]
        y_value=[]
        dummy_model.to(dev)
        for x,y in loader:
            output0.append(np.array(x).reshape(-1))
            dummy_output = dummy_model(x.to(dev))
            res = dummy_model.get_intermediate_states()
    #         input_fcs=res[0]
            out_fcs=res[-2]
            output1.append(out_fcs['input_fc_0'][0])
            output2.append(out_fcs['output_fc_0_after_dropout'][0])
            output3.append(out_fcs['output_fc_1_after_dropout'][0])
            output4.append(out_fcs['output_fc_2_after_dropout'][0])

            output5.append(out_fcs['output_fc_0_after_noise_relu'][0])
            output6.append(out_fcs['output_fc_1_after_noise_relu'][0])
            output7.append(out_fcs['output_fc_2_after_noise_relu'][0])

            y_value.append(y[0]) 
        data_layer_name=['input_layer','input_fc_0','output_fc_0_after_noise_relu',
                     'output_fc_0_after_dropout', 'output_fc_1_after_noise_relu',
                     'output_fc_1_after_dropout','output_fc_2_after_noise_relu',
                     'output_fc_2_after_dropout',
                     'y_value_corrupted']
        outputs = [output0, output1, output5, output2, output6,output3,output7,output4, y_value]

    for idx,data in enumerate(data_layer_name):
        if 0 <= idx < len(outputs):
            data_value = outputs[idx]
        with open(f'{path}/{data}_{name}.pkl', 'wb') as file: 
            pickle.dump(data_value, file)
    
    
    
    
def data_saving(path,loader,dummy_model,name,dev,type_network):
    if type_network=='AlexNet':
        dummy_output =None
#         output0=[]
        output1=[]
        output2=[]
        output3=[]
        y_value=[]
        dummy_model.to(dev)
        for x,y in loader:
#             output0.append(x.detach().cpu().numpy().reshape(-1))
            dummy_output = dummy_model(x.to(dev))
            out_fcs = dummy_model.get_intermediate_states()

            output1.append(out_fcs[1][0].cpu())
            output2.append(out_fcs[2][0].cpu())
            output3.append(out_fcs[3][0].cpu())
            y_value.append(y[0])

        data_layer_name=['after_flatten',
                         'after_relu_fc1', 
                         'after_relu_fc2',
                         'y_value_corrupted'] #'input_layer',
        outputs = [output1, output2, output3, y_value] #output0,
    if type_network=='CNN':
        dummy_output =None
        output0=[]
        output1=[]
        output2=[]
        output3=[]
        output4=[]
        y_value=[]
        dummy_model.to(dev)
        for x,y in loader:
            output0.append(x.detach().cpu().numpy().reshape(-1))
            dummy_output = dummy_model(x.to(dev))
            res = dummy_model.get_intermediate_states()
            out_fcs=res[-2]

            output1.append(out_fcs['input_fc_0'][0].cpu())
            output2.append(out_fcs['output_fc_0_after_noise_relu'][0].cpu())
            output3.append(out_fcs['output_fc_1_after_noise_relu'][0].cpu())
            output4.append(out_fcs['output_fc_2_after_noise_relu'][0].cpu())
            y_value.append(y[0])  
        data_layer_name=['input_layer','input_fc_0',
                         'output_fc_0_after_noise_relu', 
                         'output_fc_1_after_noise_relu',
                         'output_fc_2_after_noise_relu','y_value_corrupted']
        outputs = [output0, output1, output2, output3, output4, y_value]

    for idx,data in enumerate(data_layer_name):
        if 0 <= idx < len(outputs):
            data_value = outputs[idx]
        with open(f'{path}/{data}_{name}.pkl', 'wb') as file: 
            pickle.dump(data_value, file)
            
            
def data_saving_lastlayer(path,loader,dummy_model,name,dev,type_network):
    if type_network=='AlexNet':
        dummy_output =None
        output3=[]
        y_value=[]
        dummy_model.to(dev)
        for x,y in loader:
            dummy_output = dummy_model(x.to(dev))
            out_fcs = dummy_model.get_intermediate_states()
            output3.append(out_fcs[3][0].cpu())
            y_value.append(y[0])

        data_layer_name=['after_relu_fc2',
                         'y_value_corrupted']
        outputs = [output3, y_value] 
    if type_network=='CNN':
        dummy_output =None
        output4=[]
        y_value=[]
        dummy_model.to(dev)
        for x,y in loader:
            
            dummy_output = dummy_model(x.to(dev))
            res = dummy_model.get_intermediate_states()
            out_fcs=res[-2]
            output4.append(out_fcs['output_fc_2_after_noise_relu'][0].cpu())
            y_value.append(y[0])  
        data_layer_name=[ 'output_fc_2_after_noise_relu','y_value_corrupted']
        outputs = [output4, y_value]
    if type_network =='MLP':
        dummy_output =None
        output4=[]
        y_value=[]
        dummy_model.to(dev)
        for x,y in loader:
            dummy_output = dummy_model(x.to(dev))
            res = dummy_model.get_intermediate_states()
            output4.append(res[5][0].cpu())
            y_value.append(y[0])
        data_layer_name=['after_relu_fc4','y_value_corrupted']

        outputs = [output4, y_value]

    for idx,data in enumerate(data_layer_name):
        if 0 <= idx < len(outputs):
            data_value = outputs[idx]
        with open(f'{path}/{data}_{name}.pkl', 'wb') as file: 
            pickle.dump(data_value, file)
            
def loading_saving_activations(result_path,dummy_model,corrupted_train,test_loader,og_targets,dev,type_network,dropout=False):
    if dropout:
        data_saving_drop(result_path,corrupted_train,dummy_model,'train',dev,type_network)
        with open(f'{result_path}/y_value_original_train.pkl', 'wb') as file: 
            pickle.dump(og_targets, file)

        data_saving_drop(result_path,test_loader,dummy_model,'test',dev,type_network)
    else:
        #training data
        data_saving(result_path,corrupted_train,dummy_model,'train',dev,type_network)

        #saving original labels
        with open(f'{result_path}/y_value_original_train.pkl', 'wb') as file: 
            pickle.dump(og_targets, file)

        #testing data
        data_saving(result_path,test_loader,dummy_model,'test',dev,type_network)
    
def loading_saving_activations_2(result_path,dummy_model,corrupted_train,test_loader,og_targets,dev,type_network,epoch_number):
    #training data
    data_saving(result_path,corrupted_train,dummy_model,f'train_{epoch_number}',dev,type_network)

    #saving original labels
    with open(f'{result_path}/y_value_original_train.pkl', 'wb') as file: 
        pickle.dump(og_targets, file)

    #testing data
    data_saving(result_path,test_loader,dummy_model,f'test_{epoch_number}',dev,type_network)
    
def saving_activations_lastlayer(result_path,dummy_model,corrupted_train,test_loader,og_targets,dev,type_network):
    #training data
    data_saving_lastlayer(result_path,corrupted_train,dummy_model,'train',dev,type_network)

    #saving original labels
    with open(f'{result_path}/y_value_original_train.pkl', 'wb') as file: 
        pickle.dump(og_targets, file)

    #testing data
    data_saving_lastlayer(result_path,test_loader,dummy_model,'test',dev,type_network)   
    
    
    
