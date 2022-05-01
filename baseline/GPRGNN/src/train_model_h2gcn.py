import argparse
from GNN_models_h2gcn import *

import torch
import torch.nn.functional as F
from utils import random_planetoid_splits
import networkx as nx
from tqdm import tqdm
import random
import ipdb
import json
import sys

import numpy as np
import sys
from pathlib import Path
from torch_geometric.utils import from_networkx

sys.path.append('/home/syf/workspace/jupyters/Nancy/H2GCN')
from experiments.h2gcn import utils
# In[4]:
import json
import argparse


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, Flag=0, ):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing
    indices = []
    for i in range(num_classes):
        index = (torch.LongTensor(data.labels) == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    if Flag is 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        train_mask = index_to_mask(train_index, size=data.labels.shape[0])
        val_mask = index_to_mask(rest_index[:val_lb], size=data.labels.shape[0])
        test_mask = index_to_mask(
            rest_index[val_lb:], size=data.labels.shape[0])
    else:
        val_index = torch.cat([i[percls_trn:percls_trn + val_lb]
                               for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn + val_lb:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return train_mask, val_mask, test_mask


def random_planetoid_splits_syn(data, num_classes, Y, percls_trn=20, val_lb=500, Flag=0, ):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing
    indices = []
    for i in range(num_classes):
        index = (torch.LongTensor(Y) == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    if Flag is 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        train_mask = index_to_mask(train_index, size=Y.shape[0])
        val_mask = index_to_mask(rest_index[:val_lb], size=Y.shape[0])
        test_mask = index_to_mask(
            rest_index[val_lb:], size=Y.shape[0])
    else:
        val_index = torch.cat([i[percls_trn:percls_trn + val_lb]
                               for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn + val_lb:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=Y.shape[0])
        data.val_mask = index_to_mask(val_index, size=Y.shape[0])
        data.test_mask = index_to_mask(rest_index, size=Y.shape[0])
    return train_mask, val_mask, test_mask


def RunExp(args, dataset, Net, percls_trn, val_lb, RP):
    def train(model, optimizer, data, dprate, train_mask):
        model.train()
        optimizer.zero_grad()
        out = model(data)[train_mask]
        nll = F.nll_loss(out, data.y[train_mask])
        loss = nll
        loss.backward()

        optimizer.step()
        del out

    def test(model, data, train_mask, val_mask, test_mask):
        model.eval()
        logits, accs, losses, preds = model(data), [], [], []

        pred = logits[train_mask].max(1)[1]
        acc = pred.eq(data.y[train_mask]).sum().item() / train_mask.sum().item()
        loss = F.nll_loss(model(data)[train_mask], data.y[train_mask])
        preds.append(pred.detach().cpu())
        accs.append(acc)
        losses.append(loss.detach().cpu())

        pred = logits[val_mask].max(1)[1]
        acc = pred.eq(data.y[val_mask]).sum().item() / val_mask.sum().item()
        loss = F.nll_loss(model(data)[val_mask], data.y[val_mask])
        preds.append(pred.detach().cpu())
        accs.append(acc)
        losses.append(loss.detach().cpu())

        pred = logits[test_mask].max(1)[1]
        acc = pred.eq(data.y[test_mask]).sum().item() / test_mask.sum().item()
        loss = F.nll_loss(model(data)[test_mask], data.y[test_mask])
        preds.append(pred.detach().cpu())
        accs.append(acc)
        losses.append(loss.detach().cpu())

        return accs, preds, losses

    appnp_net = Net(dataset, args)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    permute_masks = random_planetoid_splits_syn
    (X, Y, num_classes, train_mask, val_mask, test_mask) = load_syn_data(name, RP)
    train_mask, val_mask, test_mask = permute_masks(dataset, num_classes, Y, percls_trn, val_lb, )

    model = appnp_net
    # model, dataset = appnp_net.to(device), dataset.to(device)

    if args.net in ['APPNP', 'GPRGNN']:
        optimizer = torch.optim.Adam([{
            'params': model.lin1.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
                'params': model.lin2.parameters(),
                'weight_decay': args.weight_decay, 'lr': args.lr
            },
            {
                'params': model.prop1.parameters(),
                'weight_decay': 0.0, 'lr': args.lr
            }
        ],
            lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    adj = dataset._sparse_data["sparse_adj"]
    features = dataset._sparse_data["features"]
    labels = dataset._dense_data["y_all"]

    feat_data = torch.FloatTensor(np.array(features.todense()))
    labels = torch.squeeze(torch.LongTensor(np.expand_dims(np.argmax(labels, 1), 1)))
    G = nx.from_scipy_sparse_matrix(adj)
    data = from_networkx(G)
    data.x = X
    data.y = Y

    for epoch in range(args.epochs):

        train(model, optimizer, data, args.dprate, train_mask)
        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data, train_mask, val_mask, test_mask)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            if args.net == 'GPRGNN':
                TEST = appnp_net.prop1.temp.clone()
                Alpha = TEST.detach().cpu().numpy()
            else:
                Alpha = args.alpha
            Gamma_0 = Alpha

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break

    return test_acc, best_val_acc, Gamma_0


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
    path = "/home/denghaoran/tmp/" + name + "_label.out"
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

    (train_mask, val_mask, test_mask) = get_whole_mask(Y, seed=round + 1)
    return X, Y, num_classes, train_mask, val_mask, test_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--train_rate', type=float, default=0.025)
    parser.add_argument('--val_rate', type=float, default=0.025)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--dprate', type=float, default=0.5)
    parser.add_argument('--C', type=int)
    parser.add_argument('--Init', type=str,
                        choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],
                        default='PPR')
    parser.add_argument('--Gamma', default=None)
    parser.add_argument('--ppnp', default='GPR_prop',
                        choices=['PPNP', 'GPR_prop'])
    parser.add_argument('--heads', default=8, type=int)
    parser.add_argument('--output_heads', default=1, type=int)

    parser.add_argument('--dataset', default='wisconsin')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--RPMAX', type=int, default=10)
    parser.add_argument('--net', type=str, choices=['GCN', 'GAT', 'APPNP', 'ChebNet', 'JKNet', 'GPRGNN'],
                        default='GPRGNN')
    data_list = ['cornell']
    args = parser.parse_args()
    for name in data_list:
        args.dataset = name
        gnn_name = args.net
        if gnn_name == 'GCN':
            Net = GCN_Net
        elif gnn_name == 'GAT':
            Net = GAT_Nett
        elif gnn_name == 'APPNP':
            Net = APPNP_Net
        elif gnn_name == 'ChebNet':
            Net = ChebNet
        elif gnn_name == 'JKNet':
            Net = GCN_JKNet
        elif gnn_name == 'GPRGNN':
            Net = GPRGNN

        dname = args.dataset
        print("#")
        print(dname)
        datasets = json.load(open('/home/srtpgroup/www_cyx/configs/dataset.json'))
        if args.dataset[:3] == "ind":
            args.dataset = args.dataset[4:]
        dataset_str = datasets[args.dataset]['dataset']
        dataset_path = datasets[args.dataset]['dataset_path'][0]
        dataset_path = '/home/srtpgroup/baseline_homo' / Path(dataset_path)
        val_size = datasets[args.dataset]['val_size']
        dataset = utils.PlanetoidData(dataset_str=dataset_str, dataset_path=dataset_path, val_size=val_size)
        # dataset = DataLoader(dname)
        # data = dataset[0]
        adj = dataset._sparse_data["sparse_adj"]
        features = dataset._sparse_data["features"]
        labels = dataset._dense_data["y_all"]

        n_nodes, n_feats = features.shape[0], features.shape[1]
        num_classes = labels.shape[-1]

        G = nx.from_scipy_sparse_matrix(adj)
        adj_lists = nx.to_dict_of_lists(G)
        adj_lists = {k: set(v) for k, v in adj_lists.items()}
        feat_data = np.array(features.todense())
        labels = np.expand_dims(np.argmax(labels, 1), 1)

        RPMAX = args.RPMAX
        Init = args.Init

        Gamma_0 = None
        alpha = args.alpha
        train_rate = args.train_rate
        val_rate = args.val_rate
        percls_trn = int(round(train_rate * dataset.labels.shape[0] / dataset.label_count.shape[0]))
        val_lb = int(round(val_rate * dataset.labels.shape[0]))
        TrueLBrate = (percls_trn * dataset.label_count.shape[0] + val_lb) / dataset.labels.shape[0]
        print('True Label rate: ', TrueLBrate)

        args.C = dataset.label_count.shape[0]
        args.Gamma = Gamma_0

        Results0 = []
        RPMAX = 10
        print(RPMAX)
        for RP in tqdm(range(RPMAX)):
            test_acc, best_val_acc, Gamma_0 = RunExp(
                args, dataset, Net, percls_trn, val_lb, RP)
            Results0.append([test_acc, best_val_acc, Gamma_0])

        test_acc_mean, val_acc_mean, _ = np.mean(Results0, axis=0) * 100
        test_acc_std = np.sqrt(np.var(Results0, axis=0)[0]) * 100
        print(f'{gnn_name} on dataset {args.dataset}, in {RPMAX} repeated experiment:')
        print(
            f'test acc mean = {test_acc_mean:.4f} \t test acc std = {test_acc_std:.4f} \t val acc mean = {val_acc_mean:.4f}')
