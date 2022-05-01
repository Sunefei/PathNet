from asyncio import proactor_events
import copy
import random
from itertools import combinations_with_replacement
import time
import matplotlib.pyplot as plt
import networkx as nx
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
from torch_geometric.nn import MessagePassing, GCNConv, GINConv, global_add_pool
from torch_geometric.utils import add_self_loops, degree, from_scipy_sparse_matrix
from sklearn.metrics import f1_score, accuracy_score
from torch.nn.parameter import Parameter
import csrgraph as cg
from ast import literal_eval
import warnings
from dataset import PlanetoidData
import tqdm
warnings.filterwarnings('ignore')

# Parameters
batch_size = 16
lr = 0.005
weight_decay = 0.0005
epochs = 1000
rounds = 10
hidden_size = [128, 128, 128, 200,
               256, 256, 64, 64
               ]
# hidden_size = {'texas': [256, 128, 64, 32, 16, 8],
#                'wisconsin': [512, 256, 128, 64, 32, 16, 8],
#                'cora': [128, 64, 32, 16],  # 150,
#                'film': [86, 64, 32, 16]
#                }
# default_nbs = 10
# num_of_walks = [40, 40, 40, 40, 40]
# walk_length = [4,  4, 4, 4, 4]
num_of_walks = [40, 40, 40, 40,
                40, 40, 40, 40
                ]
walk_length = [4,  4, 4, 4,
               4,  4, 4, 4
               ]
data_name = ['citeseer', 'wisconsin', 'cora', 'film', 'cornell',
             'chameleon', 'citeseer', 'squirrel', 'pubmed',
             ]
# data_name = ['chameleon', 'squirrel', 'pubmed', 'citeseer']
start, end = 0, 1
mode = 'pagg'
save_file_name = "syn_citeseer_pagg"
splits_file_path = "/home/syf/workspace/geom-gcn/splits/"

# Seed
# random_seed = 1
# random.seed(random_seed)
# torch.manual_seed(random_seed)
# torch.cuda.manual_seed(random_seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

# Compute Homophily

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def get_order(ratio: list, masked_index: torch.Tensor, total_node_num: int, seed: int = 1234567):
    """
    输入划分比例和原始的索引，输出对应划分的mask元组
    入参：
    ratio格式：'1-1-3'  [48,32,20]
    masked_index是索引的1维Tensor
    返回值：(train_mask,val_mask,test_mask)
    都是长度为总节点数，对应索引置True的布尔Tensor
    """
    random.seed(seed)

    masked_node_num = len(masked_index)
    shuffle_criterion = list(range(masked_node_num))
    random.shuffle(shuffle_criterion)

    # train_val_test_list=[int(i) for i in ratio.split('-')]
    train_val_test_list = ratio
    tvt_sum = sum(train_val_test_list)
    tvt_ratio_list = [i/tvt_sum for i in train_val_test_list]
    train_end_index = int(tvt_ratio_list[0]*masked_node_num)
    val_end_index = train_end_index+int(tvt_ratio_list[1]*masked_node_num)

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


def get_whole_mask(y, ratio: list = [60, 20, 20], seed: int = 1234567):
    """对整个数据集按比例进行划分[48, 32, 20]"""
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


def load_syn_data(name, round):
    path = "/data/syf/"+name+"/"+name+"_label.out"
    Y = []
    label_type = []
    with open(path, 'r') as lbl:
        for l in lbl:
            Y.append(int(l))
            if int(l) not in label_type:
                label_type.append(int(l))
    Y = torch.LongTensor(Y)
    X = torch.FloatTensor(
        [[1, 0] if i % 2 == 0 else [0, 1] for i in range(Y.shape[0])])
    num_classes = len(label_type)

    (train_mask, val_mask, test_mask) = get_whole_mask(Y, seed=round+1)
    return X, Y, num_classes, train_mask, val_mask, test_mask


def load_data_ranked(name):
    datasets = json.load(
        open("/home/syf/workspace/jupyters/configs/dataset.json"))
    dataset_run = datasets[name]["dataset"]
    dataset_path = datasets[name]["dataset_path"][0]
    dataset_path = "/home/syf/workspace/jupyters" / Path(dataset_path)
    val_size = datasets[name]["val_size"]

    dataset = PlanetoidData(
        dataset_str=dataset_run, dataset_path=dataset_path, val_size=val_size
    )

    adj = dataset._sparse_data["sparse_adj"]
    features = dataset._sparse_data["features"]
    # labels = dataset._dense_data["y_all"]

    # n_nodes, n_feats = features.shape[0], features.shape[1]
    # num_classes = labels.shape[-1]

    # G = cg.csrgraph(adj, threads=0)
    # G.set_threads(0)  # number of threads to use. 0 is full use
    edge = from_scipy_sparse_matrix(adj)[0]  # indices + edge_weight
    # X = torch.tensor(features.todense(), dtype=torch.float)
    # label = torch.tensor(np.argmax(labels, 1), dtype=torch.long)

    edge_index = torch.LongTensor(edge)
    # sparse_ones = torch.ones(edge.shape[-1], dtype=torch.long)
    # sparse_A = torch.sparse.LongTensor(
    #     sparse_edges, sparse_ones, torch.Size([label.shape[0], label.shape[0]]))
    # A = sparse_A.to_dense()
    # AX = torch.mm(A + torch.eye(X.shape[0]), X)  # self-loops included
    return edge_index
    # return (X, G, label, num_classes, datasets)


class MLP(torch.nn.Module):
    def __init__(self, feature_length, hidden_size, out_size):
        super(MLP, self).__init__()
        self.feature_length, self.hidden_size, self.out_size = feature_length, hidden_size, out_size
        self.fc1 = torch.nn.Linear(feature_length, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, out_size)

    def forward(self, x, edge_index):
        x = x.view(-1, self.feature_length)
        # print(x, x.shape)
        # print(edge_index, edge_index.shape)
        x = F.dropout(x, p=0.9, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.9, training=self.training)
        x = self.fc2(x)
        return x


class GAT(torch.nn.Module):
    def __init__(self, feature_length, hidden_size, out_size):
        super(GAT, self).__init__()
        self.conv1 = GCNConv(feature_length, 16, cached=True,
                             normalize=True)
        self.conv2 = GCNConv(16, out_size, cached=True,
                             normalize=True)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.9, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.9, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GIN(torch.nn.Module):
    def __init__(self, in_channels, dim, out_channels):
        super(GIN, self).__init__()

        self.conv1 = GINConv(
            nn.Sequential(nn.Linear(in_channels, dim), nn.BatchNorm1d(dim), nn.ReLU(),
                          nn.Linear(dim, dim), nn.ReLU()))

        self.conv2 = GINConv(
            nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
                          nn.Linear(dim, dim), nn.ReLU()))

        # self.conv3 = GINConv(
        #     Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
        #                Linear(dim, dim), ReLU()))

        # self.conv4 = GINConv(
        #     Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
        #                Linear(dim, dim), ReLU()))

        # self.conv5 = GINConv(
        #     Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
        #                Linear(dim, dim), ReLU()))

        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        # x = self.conv3(x, edge_index)
        # x = self.conv4(x, edge_index)
        # x = self.conv5(x, edge_index)
        # x = global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        # return F.log_softmax(x, dim=-1)
        return x


class ParMLP(MessagePassing):
    def __init__(self, feature_length, hidden_size, out_size, node_num, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super(ParMLP, self).__init__()
        self.feature_length, self.hidden_size, self.out_size, self.node_num \
            = feature_length, hidden_size, out_size, node_num

        self.fc0 = torch.nn.Linear(feature_length, hidden_size)
        # self.fc1 = torch.nn.Linear(feature_length, hidden_size)  # 收第一层输入
        self.LSTM = nn.LSTM(hidden_size, hidden_size)
        self.fc2 = torch.nn.Linear(2*hidden_size, out_size)  # 收第二层输入
        # self.fc3 = torch.nn.Linear(hidden_size, out_size)
        torch.nn.init.xavier_uniform_(self.fc0.weight)
        # torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        # torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, X, neis, num_w, walk_len, indices):
        split = sum(indices)
        ego = self.fc0(X)
        neis = neis.to(device)
        nei = ego[neis].view(split*num_w, walk_len, -1)

        nei = nei.transpose(0, 1)
        nei = torch.flip(nei, dims=[0])  # 最后节点是ego
        nei, (h_n, c_n) = self.LSTM(nei)
        h_n = h_n.transpose(0, 1).view(
            split, num_w, -1)  # [V, num_of_walks, H]

        h_n = torch.mean(h_n, dim=1)

        layer1 = torch.cat((ego[indices], h_n), dim=1)  # [V, 2*H]
        # layer1 = F.relu(self.fc2(layer1))
        layer1 = F.dropout(layer1, p=0.5, training=self.training)
        dout = self.fc2(layer1)
        return dout


class PAGG(MessagePassing):
    def __init__(self, feature_length, hidden_size, out_size, node_num, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super(PAGG, self).__init__()
        self.feature_length, self.hidden_size, self.out_size, self.node_num \
            = feature_length, hidden_size, out_size, node_num

        self.fc0 = torch.nn.Linear(feature_length, hidden_size)
        # self.fc1 = torch.nn.Linear(feature_length, hidden_size)  # 收第一层输入
        self.RNN = nn.RNN(hidden_size, hidden_size)
        self.fc2 = torch.nn.Linear(2*hidden_size, out_size)  # 收第二层输入
        self.nei0 = torch.nn.Linear(hidden_size, hidden_size)
        self.nei1 = torch.nn.Linear(hidden_size, hidden_size)
        self.nei2 = torch.nn.Linear(hidden_size, hidden_size)
        self.nei3 = torch.nn.Linear(hidden_size, hidden_size)
        # self.fc3 = torch.nn.Linear(hidden_size, out_size)
        torch.nn.init.xavier_uniform_(self.fc0.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.nei0.weight)
        torch.nn.init.xavier_uniform_(self.nei1.weight)
        torch.nn.init.xavier_uniform_(self.nei2.weight)
        torch.nn.init.xavier_uniform_(self.nei3.weight)
        # self.RNN_dict = {0: self.nei0,
        #                  1: self.nei1,
        #                  2: self.nei2,
        #                  3: self.nei3,
        #                  }

    def forward(self, X, neis, num_w, walk_len, indices, layer_type, indxx):
        split = sum(indices)
        # X = X.to(device)
        X = self.fc0(X)
        neis = neis.to(device)
        # layer_type = layer_type.to(device)
        # torch.Size([3480*4, 128])
        nei = X[neis].view(split*num_w*walk_len, self.hidden_size)
        # nei = nei.transpose(0, 1)  # (walk_len, split*num_w, self.hidden_size)
        # nei = torch.flip(nei, dims=[0])  # 最后节点是ego
        layer_type = layer_type.view(split*num_w*walk_len).to(device)

        nei0 = self.nei0(nei)
        nei1 = self.nei1(nei)
        nei2 = self.nei2(nei)
        nei3 = self.nei3(nei)
        neis_cat = torch.stack((nei0, nei1, nei2, nei3), dim=1)

        nei = neis_cat[indxx, layer_type].view(
            split*num_w, walk_len, self.hidden_size).transpose(0, 1)
        # print(nei.shape)  # torch.Size([4, 3480, 128])
        nei = F.dropout(nei, p=0.9, training=self.training)
        nei, h_n = self.RNN(nei)
        h_n = h_n.transpose(0, 1).view(
            split, num_w, -1)  # [V, num_of_walks, H]

        h_n = torch.mean(h_n, dim=1)
        ego = X[indices]
        layer1 = torch.cat((ego, h_n), dim=1)  # [V, 2*H]
        # layer1 = F.relu(self.fc2(layer1))
        layer1 = F.dropout(layer1, p=0.9, training=self.training)
        dout = self.fc2(layer1)
        return dout


def train_fixed_indices(X, edge, Y, num_classes, mode, data_name, train_indices, val_indices, test_indices, num_w, hid_size, walk_len, walks, path_type_all):
    feature_length = X.shape[-1]
    node_num = Y.shape[0]
    # Construct the model
    if mode == 'our':
        predictor = RNNMean(feature_length, hid_size,
                            num_classes, node_num).to(device)
    elif mode == 'pagg':
        predictor = PAGG(feature_length, hid_size,
                         num_classes, node_num).to(device)
    elif mode == 'pagg_sum':
        predictor = PAGG_sum(feature_length, hid_size,
                             num_classes, node_num).to(device)
    elif mode == 'mlp':
        predictor = MLP(feature_length, hid_size, num_classes).to(device)
    elif mode == 'gat':
        predictor = GAT(feature_length, hid_size, num_classes).to(device)

    optimizer = torch.optim.Adam(
        predictor.parameters(), lr=lr, weight_decay=weight_decay)
    lossfunc = torch.nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=5, eta_min=4e-08)

    # prep data
    X = X.to(device)
    Y = Y.to(device)

    # Start training
    max_val_acc = 0
    val_acc = 0  # validation
    test_acc = 0
    train_bar = tqdm.tqdm(range(epochs), dynamic_ncols=True, unit='step')
    loss_collector, val_acc_coll = [], []
    neis_all = torch.tensor(walks, dtype=torch.long).view(
        1000, node_num, -1)
    path_type_all = torch.tensor(path_type_all, dtype=torch.long).view(  # 整体文件还没重新生成
        1000, node_num, num_w, walk_len)  # -torch.ones(epochs, node_num, num_w, walk_len).long()

    if mode != "pagg":
        for epoch in train_bar:
            predictor.train()
            y_hat = predictor(X, edge.to(device))  # transductive!!
            # neis[train_indices] = neis[train_indices].to('cpu')
            loss = lossfunc(y_hat[train_indices], Y[train_indices])
            loss_collector.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                predictor.eval()
                y_hat = F.log_softmax(
                    predictor(X, edge.to(device)), dim=1)
                y_hat_ = y_hat[val_indices].cpu().max(1)[1]
                val_acc = accuracy_score(Y[val_indices].cpu(), y_hat_)
                val_acc_coll.append(val_acc)

                if val_acc > max_val_acc:
                    max_val_acc = val_acc
                    # print("Save Model.")
                    torch.save(predictor.state_dict(),
                               "/data/syf/models/" + save_file_name + ".pth")
                    y_hat_ = y_hat[test_indices].cpu().max(1)[1]
                    test_acc = accuracy_score(Y[test_indices].cpu(), y_hat_)
                train_bar.set_postfix(
                    data=data_name, val_acc=val_acc, test_acc=test_acc)

    else:
        for epoch in train_bar:
            # walks = []
            # path_file = "/data/syf/rw/{}_{}_{}_hr_{:04d}.txt".format(
            #     name, num_w, walk_len, epoch)
            # with open(path_file, "r") as p:
            #     for line in p:
            #         walks.append(list(map(int, line[1:-2].split(","))))
            # print("walks", len(walks))
            # neis = torch.tensor(walks, dtype=torch.long)
            # neis = neis.view(node_num, num_w, walk_len)
            neis = neis_all[epoch]
            path_type = path_type_all[epoch]
            predictor.train()
            # neis[train_indices] = neis[train_indices]
            indxx = torch.arange(
                sum(train_indices)*num_w*walk_len, dtype=torch.long, device=device)
            time2 = time.time()
            y_hat = predictor(X, neis[train_indices],
                              num_w, walk_len, train_indices, path_type[train_indices], indxx)  # transductive!! path_type[train_indices]
            loss = lossfunc(y_hat, Y[train_indices].to(device))
            loss_collector.append(loss)

            # scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            last_test_acc = 0
            with torch.no_grad():
                predictor.eval()
                # neis[val_indices] = neis[val_indices].to(device)
                indxx = torch.arange(
                    sum(val_indices)*num_w*walk_len, dtype=torch.long, device=device)
                y_hat = F.log_softmax(
                    predictor(X, neis[val_indices], num_w, walk_len, val_indices, path_type[val_indices], indxx), dim=1)
                # neis[val_indices] = neis[val_indices].to('cpu')
                # total_val_loss = lossfunc(y_hat[val_indices], Y[val_indices]).item()
                y_hat_ = y_hat.cpu().max(1)[1]
                val_acc = accuracy_score(Y[val_indices].cpu(), y_hat_)
                # print(data_name, "val_acc", val_acc)
                val_acc_coll.append(val_acc)

                if val_acc > max_val_acc:
                    max_val_acc = val_acc
                    # print("Save Model.")
                    torch.save(predictor.state_dict(),
                               "models/" + save_file_name + ".pth")
                    # test_acc = 0
                    # for k in range(test_rw_round):
                    #     ch = random.choice(list(range(epochs)))
                    #     neis = neis_all[ch]
                    # neis[test_indices] = neis[test_indices].to(device)
                    indxx = torch.arange(
                        sum(test_indices)*num_w*walk_len, dtype=torch.long, device=device)
                    y_hat = F.log_softmax(
                        predictor(X, neis[test_indices], num_w, walk_len, test_indices, path_type[test_indices], indxx), dim=1)
                    # neis[test_indices] = neis[test_indices].to('cpu')
                    y_hat_ = y_hat.cpu().max(1)[1]
                    test_acc = accuracy_score(Y[test_indices].cpu(), y_hat_)
                    #     test_acc += test_tmp
                    # test_acc /= test_rw_round
                train_bar.set_postfix(
                    data=data_name, val_acc=val_acc, test_acc=test_acc)
    return test_acc, loss_collector, val_acc_coll  # val_acc is a list


file = open("results/" + save_file_name + ".txt", "a")
for k in range(start, end):
    # for name in [
    #     # "film",  # 128
    #     "cora",  # 200
    #     "pubmed",  # 128!
    #     "citeseer",  # 200
    #     # "squirrel",  # 128
    #     # "chameleon",
    #     # "texas",
    #     # "cornell",
    #     # "wisconsin",
    # ]:

    name = data_name[k]
    hidden=hidden_size[k]
    walks = []
    path_type = []
    edge = None
    if name == 'citeseer':
        edge = load_data_ranked(name)
    path_file = "/data/syf/rw/" + name + "_" + \
        str(num_of_walks[k]) + "_" + str(walk_length[k]) + '_m.txt'
    with open(path_file, "r") as p:
        for line in p:
            info = list(map(int, line[1:-2].split(",")))
            walks.append(info[:walk_length[k]])
            path_type.append(info[walk_length[k]:])
    # for hidden in hidden_size[name]:
    avg_acc_gcn, avg_acc_cat, std_acc_gcn, std_acc_cat, avg_acc_mlp, std_acc_mlp = 0, 0, 0, 0, 0, 0
    acc_mlps, acc_gcns, acc_cats = [], [], []
    # train_l,val_ac=np.empty(epochs),np.empty(epochs)
    train_l, val_ac = [], []

    # (X, G, Y, num_classes, datasets) = load_data_ranked(name)
    for i in range(rounds):
        print('round', i)

        (X, Y, num_classes, train_mask, val_mask,
            test_mask) = load_syn_data(name, i)
        # dataset_run = datasets[name]["dataset"]
        # dataset_path = datasets[name]["dataset_path"][i]
        # dataset_path = "/home/syf/workspace/jupyters" / \
        #     Path(dataset_path)
        # val_size = datasets[name]["val_size"]

        # dataset = PlanetoidData(
        #     dataset_str=dataset_run, dataset_path=dataset_path, val_size=val_size
        # )

        # train_mask = dataset._dense_data["train_mask"]
        # val_mask = dataset._dense_data["val_mask"]
        # test_mask = dataset._dense_data["test_mask"]

        test_cat, train_loss, val_acc = train_fixed_indices(
            X, edge, Y, num_classes, mode, name, train_mask, val_mask, test_mask, num_of_walks[k], hidden, walk_length[k], walks, path_type)
        # acc_gcns.append(test_gcn)
        acc_cats.append(test_cat)
        # acc_mlps.append(test_mlp)
        train_l.append(train_loss)
        val_ac.append(val_acc)
    # for i in range(rounds):
    #     plt.subplot(4, 4, i+1)
    #     plt.plot(train_l[i])
    # plt.savefig("/home/syf/workspace/results/train_loss_" +
    #             name+'_'+".png", dpi=200)
    # plt.clf()
    # for i in range(rounds):
    #     plt.subplot(4, 4, i+1)
    #     plt.plot(val_ac[i])
    # plt.savefig("/home/syf/workspace/results/val_acc_" +
    #             name+'_'+".png", dpi=200)
    # plt.clf()

    #     train_l+=train_loss
    #     val_ac+=val_acc
    # train_l/=rounds
    # val_ac/=rounds

    # avg_acc_gcn = sum(acc_gcns) / rounds
    avg_acc_cat = sum(acc_cats) / rounds
    # avg_acc_mlp = sum(acc_mlps) / rounds
    # std_acc_gcn = np.std(np.array(acc_gcns))
    std_acc_cat = np.std(np.array(acc_cats))
    # std_acc_mlp = np.std(np.array(acc_mlps))

    # print("GCN Accuracy for {}: {:.4f} ± {:.4f}\n".format(
    #     name, avg_acc_gcn, std_acc_gcn), file=file)
    # print("MLP Accuracy for {}: {:.4f} ± {:.4f}\n".format(
    #     name, avg_acc_mlp, std_acc_mlp), file=file)
    print(name+"_"+str(num_of_walks[k])+"_" +
            str(walk_length[k])+'_'+str(hidden)+"\n")
    print("CAT Accuracy for {}: {:.4f} ± {:.4f}\n".format(
        name, avg_acc_cat, std_acc_cat))
    print(name+"_"+str(num_of_walks[k])+"_"+str(walk_length[k]
                                                )+'_'+str(hidden)+"\n", file=file)
    print("CAT Accuracy for {}: {:.4f} ± {:.4f}\n".format(
        name, avg_acc_cat, std_acc_cat), file=file)
