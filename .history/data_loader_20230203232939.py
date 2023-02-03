from asyncio import proactor_events
import copy
import random
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pandas as pd
import json
import csv
import sys
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import add_self_loops, degree, from_scipy_sparse_matrix
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from torch.nn.parameter import Parameter
# import csrgraph as cg
from ast import literal_eval
import warnings
import time
import gc
from dataset import PlanetoidData  # Code in the outermost folder
import tqdm

import argparse
import pickle
import gzip

warnings.filterwarnings('ignore')

json_path='./'  # public
# json_path = '/home/syf/workspace/jupyters/configs/'


def load_data_ranked(name):
    '''
    Load data for Cora, Cornell, Pubmed and Citeseer
    '''
    datasets = json.load(
        open(json_path + "dataset.json"))
    dataset_run = datasets[name]["dataset"]
    dataset_path = datasets[name]["dataset_path"][0]
    # dataset_path = "dataset" / Path(dataset_path)
    val_size = datasets[name]["val_size"]

    dataset = PlanetoidData(
        dataset_str=dataset_run, dataset_path=dataset_path, val_size=val_size
    )

    # adj = dataset._sparse_data["sparse_adj"]
    features = dataset._sparse_data["features"]
    labels = dataset._dense_data["y_all"]

    # n_nodes, n_feats = features.shape[0], features.shape[1]
    num_classes = labels.shape[-1]

    # G = cg.csrgraph(adj, threads=0)
    # G.set_threads(0)  # number of threads to use. 0 is full use
    # edge = nx.from_scipy_sparse_matrix(adj)  # indices + edge_weight
    X = torch.tensor(features.todense(), dtype=torch.float)
    label = torch.tensor(np.argmax(labels, 1), dtype=torch.long)
    return (X, label, num_classes, datasets)


def get_order(ratio: list, masked_index: torch.Tensor, total_node_num: int, seed: int = 1234567):
    '''
    work for "get_whole_mask"
    '''
    random.seed(seed)

    masked_node_num = len(masked_index)
    shuffle_criterion = list(range(masked_node_num))
    random.shuffle(shuffle_criterion)

    # train_val_test_list=[int(i) for i in ratio.split('-')]
    train_val_test_list = ratio
    tvt_sum = sum(train_val_test_list)
    tvt_ratio_list = [i / tvt_sum for i in train_val_test_list]
    train_end_index = int(tvt_ratio_list[0] * masked_node_num)
    val_end_index = train_end_index + int(tvt_ratio_list[1] * masked_node_num)

    train_mask_index = shuffle_criterion[:train_end_index]
    val_mask_index = shuffle_criterion[train_end_index:val_end_index]
    test_mask_index = shuffle_criterion[val_end_index:]

    train_mask = torch.zeros(total_node_num, dtype=torch.bool)
    train_mask[masked_index[train_mask_index]] = True
    val_mask = torch.zeros(total_node_num, dtype=torch.bool)
    val_mask[masked_index[val_mask_index]] = True
    test_mask = torch.zeros(total_node_num, dtype=torch.bool)
    test_mask[masked_index[test_mask_index]] = True

    return (train_mask, val_mask, test_mask)


def get_whole_mask(y, ratio: list = [48, 32, 20], seed: int = 1234567):
    '''
    work for "load_data", random_spilt at [48, 32, 20] ratio
    '''
    y_have_label_mask = y != -1
    total_node_num = len(y)
    y_index_tensor = torch.tensor(list(range(total_node_num)), dtype=int)
    masked_index = y_index_tensor[y_have_label_mask]
    while True:
        (train_mask, val_mask, test_mask) = get_order(
            ratio, masked_index, total_node_num, seed)
        # if check_train_containing(train_mask,y):
        return (train_mask, val_mask, test_mask)
        # else:
        #     seed+=1


def load_data(dataset_name, round, data_root="./other_data"):
    '''
    Load data for Nba, Electronics, Bgp
    '''
    numpy_x = np.load(data_root + '/' + dataset_name + '/x.npy')
    x = torch.from_numpy(numpy_x).to(torch.float)
    numpy_y = np.load(data_root + '/' + dataset_name + '/y.npy')
    y = torch.from_numpy(numpy_y).to(torch.long)
    # numpy_edge_index = np.load(data_root+'/'+dataset_name+'/edge_index.npy')
    # edge_index = torch.from_numpy(numpy_edge_index).to(torch.long)
    (train_mask, val_mask, test_mask) = get_whole_mask(y, seed=round + 1)

    lbl_set = []
    for lbl in y:
        if lbl not in lbl_set:
            lbl_set.append(lbl)
    num_classes = len(lbl_set)

    return x, y, num_classes, train_mask, val_mask, test_mask
