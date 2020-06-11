# online-deep-learning-for-non-stationary-dataset


![](https://img.shields.io/badge/linux-ubuntu-red.svg)
![](https://img.shields.io/badge/Mac-OS-red.svg)


![](https://img.shields.io/badge/python-3.6-green.svg)


![](https://img.shields.io/badge/matplotlib-3.0.0-blue.svg)
![](https://img.shields.io/badge/numpy-1.15.2-blue.svg)
![](https://img.shields.io/badge/pandas-0.23.3-blue.svg)
![](https://img.shields.io/badge/scipy-1.1.0-blue.svg)
![](https://img.shields.io/badge/seaborn-0.9.0-blue.svg)
![](https://img.shields.io/badge/sklearn-0.20.1-blue.svg)
![](https://img.shields.io/badge/tensorflow-1.13.0-blue.svg)


# Algorithm Implemented
1. Elastic Weight Consolidation(EWC)
2. Online SGD 
3. Tiny Episodic Memories (Tiny)
4. Variational continual learning (VCL)
5. Averaged GEM (AGEM)

# Dataset
1. MNIST  
    - Permuted
    - Occlusion
    - Darker
    - Brighter
    - Blurring
    - Noisy
    - Original
2. Steel Defect Detection  
    https://www.kaggle.com/c/severstal-steel-defect-detection
3. The Street View House Numbers (SVHN) Dataset  
    http://ufldl.stanford.edu/housenumbers/


# Example Commands
### Single Run on EWC
```
python general_main.py --data cifar100 --model EWC --learning_rate 0.001 --optimizer Adam --arch resnet18_s --num_runs 1 --lambda_ 100 --train_scheme incre --num_tasks 10 --head single
```

### Single Run on VCL
```
python general_main.py --data c --model VCL --learning_rate 0.001 --optimizer Adam
```

### Single run on Tiny Memory with Reduced ResNet18

```
python general_main.py --data cifar100 --model EWC --learning_rate 0.001 --optimizer Adam --arch resnet18_s
```

### Single run on A-GEM with Reduced ResNet18

```
python general_main.py --data cifar100 --model AGEM --learning_rate 0.003 --num_tasks 10 --optimizer Adam --arch resnet18_s --head multi --eps_mem_batch 256 --mem_size 13
```

### New Instance Non-stationary run
```
python general_main.py --data cifar-NI --model Tiny --learning_rate 0.003 --num_tasks 10 --optimizer Adam --arch resnet18_s --head single --eps_mem_batch 256 --mem_size 13
```

### Maximally Interfered Retrieval
```
python general_main.py --data cifar10 --model MIR --learning_rate 0.001 --optimizer Adam --arch resnet18_s  --train_scheme full --num_tasks 5 --head single --batch 10 --subsample 50 --eps_mem_batch 10 --mem_size 50
```

### Gradient based Sample Selection - Greedy
```
python general_main.py --data cifar10 --model GSS_Greedy --learning_rate 0.01 --optimizer Adam --arch resnet18_s --num_tasks 5 --head single --eps_mem_batch 10 --mem_size 20 --num_sample_grad 10 --num_iter 1
```

### Meta Experience Replay
```
python general_main.py --data cifar10 --model MER --learning_rate 0.003 --num_tasks 5 --optimizer Adam --arch resnet18_s --head single --num_runs 5 --train_scheme full --mem_size 100 --batch 10 --eps_mem_batch 10 --gamma 1.0 --beta 0.03 --num_iter_batch 5
```


