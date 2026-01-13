# Decoding Generalization from Memorization in Deep Neural Networks
Paper - [Decoding Generalization from Memorization in Deep Neural Networks](https://arxiv.org/pdf/2501.14687) (TMLR, 2026)
[ARXIV Version](https://arxiv.org/pdf/2501.14687)

## Contents

- [Abstract](#abstract)
- [Overview](#overview)
- [Installation Guide](#installation-guide)
- [Citation](#citation)
- [License](./LICENSE)

# Abstract

Overparameterized deep networks that generalize well have been key to the dramatic success of deep learning in recent years. The reasons for their remarkable ability to generalize are not well understood yet. 
When class labels in the training set are shuffled to varying degrees, it is known that deep networks can still reach perfect training accuracy at the detriment of generalization to true labels -- a phenomenon that has been called memorization. It has, however, been unclear why the poor generalization to true labels that accompanies such memorization, comes about. One possibility is that during training, all layers of the network irretrievably re-organize their representations in a manner that makes generalization to true labels difficult. The other possibility is that one or more layers of the trained network retain significantly more latent ability to generalize to true labels, but the network somehow “chooses” to readout in a manner that is detrimental to generalization to true labels. Here, we provide evidence for the latter possibility by demonstrating, empirically, that such models possess information in their representations for substantially-improved generalization to true labels. Furthermore, such  abilities can be easily decoded from the internals of the trained model, and we build a technique to do so. We demonstrate results on multiple models trained with standard datasets. 


# Overview
This repository provides python implementation of the algorithms described in the paper.

Pytorch implementation of CNN, AlexNet and ResNet18 models:
* Training models with different corruption degrees
* Experiments related to MASC implemented on layer wise outputs of the memorized models (i.e., subspaces with corrupted labels and true labels)
* Experiment where MASC is used on layer wise outputs of the generalized model (i.e., subspaces with corrupted labels)
* Experiment to retrain the models with relabeling performed.
* Experiment on only ResNet-18 model trained on CIFAR-10 is available in Compare_experiments/Modern_backbones.
* Experiment for training with different seeds and various PCA thresholds is available in Compare_experiments/sensitivity_ablation_studies.
* Experiment for comparing with different probes are available in Compare_experiments/stronger_baselines.

# Repo Contents
For training CNN models and Alexnet models we have used pytorch library.

- [MASC](./MASC):  Minimum Angle Subspace Classifer code. Pytorch code for different models and datasets used in the paper is available here.
- [pytorch_training](./pytorch_training.py): Pytorch code to train CNN model with CIFAR-10, Fashion-MNIST, MNIST datasets, AlexNet model with Tiny ImageNet, CIFAR-100 datasets ResNet-18 mode with CIFAR-10 dataset with different corruption degrees (i.e., 0.0,0.2,0.4,0.6,0.8,1.0).
- [pytorch_MASC_all](./pytorch_MASC_all.py): Pytorch code to run experiments related to MASC on memorized models with subspaces corresponding to corrupted and true labels. And code related to MASC on generalized models with corrupted subspaces on CNN and AlexNet Models.
- [pytorch_retrain_early](./pytorch_retrain_early.py): Pytorch code for retraining models using MASC predictions.
- [Compare_experiments](./Compare_experiments): This folder has experiments on only ResNet-18 model trained on CIFAR-10, experiment for training with different seeds and various PCA thresholds and  experiment for comparing with different probes. Each folder has its own instructions.txt file to understand how to run the experiments 

# Installation Guide


 * Clone or download the current version from project page in github or from git command line:
```
git clone git@github.com:simranketha/MASC_DNN.git
```

 * Install the related packages:

```
conda create --name pytorch_new python==3.8.10
conda init bash
source .bashrc
conda activate pytorch_new 
pip install scikit-learn tqdm
pip install pandas

for gpu-version: pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 
for cpu-version: pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu 
```

- Check [instruction.txt](./instruction.txt) to run the experiments.


# Citation

Please cite our paper:

```
@article{ketha2025decoding,
  title={Decoding Generalization from Memorization in Deep Neural Networks},
  author={Ketha, Simran and Ramaswamy, Venkatakrishnan},
  journal={arXiv preprint arXiv:2501.14687},
  year={2025}
}
```
