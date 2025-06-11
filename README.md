# Decoding Generalization from Memorization in Deep Neural Networks
[ARXIV PAPER](https://arxiv.org/pdf/2501.14687)
  
## Contents

- [Abstract](#abstract)
- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Citation](#citation)
- [License](./LICENSE)

# Abstract

Overparameterized Deep Neural Networks that generalize well have been key to the dramatic success of Deep Learning in recent years. The reasons for their remarkable ability to generalize are not well understood yet. It has also been known that deep networks possess the ability to memorize training data, as evidenced by perfect or high training accuracies on models trained with corrupted data that have class labels shuffled to varying degrees. Concomitantly, such models are known to generalize poorly, i.e. they suffer from poor test accuracies, due to which it is thought that the act of memorizing substantially degrades the ability to generalize. It has, however, been unclear why the poor generalization that accompanies such memorization, comes about. One possibility is that in the process of training with corrupted data, the layers of the network irretrievably reorganize their representations in a manner that makes generalization difficult. The other possibility is that the network retains significant ability to generalize, but the trained network somehow chooses to readout in a manner that is detrimental to generalization. Here, we provide evidence for the latter possibility by demonstrating, empirically, that such models possess information in their representations for substantially improved generalization, even in the face of memorization. Furthermore, such generalization abilities can be easily decoded from the internals of the trained model, and we build a technique to do so from the outputs of specific layers of the network. We demonstrate results on multiple models trained with a number of standard datasets.

# Overview
This repository provides python implementation of the algorithms described in the paper.


# Sytem requirements
We have used tensorflow package for MLP models and pytorch package for CNN and AlexNet models.

# Installation Guide


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
