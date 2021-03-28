# Online Class-Incremental Continual Learning with Adversarial Shapley Value


![](https://img.shields.io/badge/linux-ubuntu-red.svg)
![](https://img.shields.io/badge/Mac-OS-red.svg)


![](https://img.shields.io/badge/python-3.6-green.svg)

![](https://img.shields.io/badge/matplotlib-3.0.0-blue.svg)
![](https://img.shields.io/badge/numpy-1.15.2-blue.svg)
![](https://img.shields.io/badge/pandas-0.23.3-blue.svg)
![](https://img.shields.io/badge/scipy-1.1.0-blue.svg)
![](https://img.shields.io/badge/seaborn-0.9.0-blue.svg)
![](https://img.shields.io/badge/sklearn-0.20.1-blue.svg)
![](https://img.shields.io/badge/tensorflow-1.14.0-blue.svg)

## Notes !!
- This repository contains the TensorFlow implementation of ASER and other baselines. The results in the paper can be reproduced by following the instructions below.
- PyTorch implementation of ASER and more baselines can be found in [this repository](https://github.com/RaptorMai/online-continual-learning). Note that the PyTorch version is more efficient than the original TensorFlow implementation and has better performance.


## Requirements

```sh
conda env create -f environment.yml
```



## Data 

- CIFAR10 & CIFAR100 will be downloaded during the first run
- Mini-ImageNet: Download from https://www.kaggle.com/whitemoon/miniimagenet/download , and place in Data/miniimagenet/



## Running Experiments

* ASER = Adversarial Shapley Value Experience Replay
* AGEM = Averaged Gradient Episodic Memory
* ER = Experience Replay
* EWC = Elastic Weight Consolidation
* MIR = Maximally Interfered Retrieval
* GSS = Gradient-Based Sample Selection



To reproduce the result in the paper:

```sh
source reproduce.sh
```

