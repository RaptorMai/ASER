from Models.EWC import EWC
from Models.Tiny_mem import Tiny_mem
from Models.AGEM import AGEM
from Models.GSS_Greedy import GSS_Greedy
from Models.svknn import SVKNN
from Data.data import *
import tensorflow as tf

models = {
    "EWC": EWC,
    "Tiny": Tiny_mem,
    "AGEM": AGEM,
    'MIR': Tiny_mem,
    "GSS_Greedy": GSS_Greedy,
    'SVKNN': SVKNN
}

optimizers = {
    "GD": tf.train.GradientDescentOptimizer,
    "Adam": tf.train.AdamOptimizer
}

input_sizes = {
    "mnist" :[784],
    'cifar100': [32, 32, 3],
    'cifar10': [32, 32, 3],
    'cifar10-NI': [32, 32, 3],
    'steel': [128, 128, 1],
    'steel-NI': [128, 128, 1],
    'split_mnist': [784],
    'miniimagenet': [84, 84, 3],
    'cub': [224, 224, 3],
    'awa': [224, 224, 3]
}

output_sizes = {
    "mnist": 10,
    'cifar100': 100,
    'cifar10':10,
    'cifar10-NI': 10,
    'steel': 5,
    'steel-NI': 5,
    'split_mnist': 10,
    'miniimagenet': 100,
    'cub': 200,
    'awa': 50
}

non_stationary = {
    "Permuted": Permuted,
    "Occlusion": Occlusion,
    "Darker": Darker,
    "Brighter": Brighter,
    "Blurring": Blurring,
    "Noisy": Noisy,
    'Original': Data,
    'Step': Imbalance_step,
    'Linear': Imbalance_linear
}
