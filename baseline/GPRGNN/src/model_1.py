#!/usr/bin/env python
# coding: utf-8

# In[33]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
import json
import networkx as nx
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

from encoders import Encoder, STDEncoder
from aggregators import MeanAggregator, MeanStdAggregator, MeanStdSelfAggregator

import sys
import os.path

sys.path.append('/home/syf/workspace/jupyters/Nancy/H2GCN')
from experiments.h2gcn import utils

# In[39]:


import argparse

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='normal', help='std or normal')
parser.add_argument('--homomode', default='local', help='local or global')
parser.add_argument('--self', action='store_true', default=False,
                    help='Do not use node own features.')
parser.add_argument('--no-concat', action='store_true', default=False,
                    help='Do not concat neighbors features.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args(args=['--dataset', 'ind.citeseer', '--hidden', '64', '--lr', '0.1', '--epochs', '500'])
# args = parser.parse_args()
args.cuda = True

print(args)

device = torch.device('cuda:0')

# In[40]:


# datasets = json.load(open('/home/syf/workspace/jupyters/Nancy/graphsage-simple/configs/dataset.json'))
datasets = json.load(open('/home/xushizhe/dataset.json'))
if args.dataset[:3] == "ind":
    args.dataset = args.dataset[4:]

dataset_str = datasets[args.dataset]['dataset']
dataset_path = datasets[args.dataset]['dataset_path'][0]
dataset_path = '/home/syf/workspace/jupyters/Nancy/H2GCN' / Path(dataset_path)
val_size = datasets[args.dataset]['val_size']

dataset = utils.PlanetoidData(dataset_str=dataset_str, dataset_path=dataset_path, val_size=val_size)

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


# In[41]:


class SupervisedGraphSage(nn.Module):

    def __init__(self, feat_data, num_classes):
        super(SupervisedGraphSage, self).__init__()
        features = nn.Embedding(n_nodes, n_feats)
        if args.cuda:
            features.weight = nn.Parameter(torch.FloatTensor(feat_data).to(device), requires_grad=False)
        else:
            features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

        agg1 = MeanAggregator(features, cuda=args.cuda)
        enc1 = Encoder(features, n_feats, args.hidden, adj_lists, agg1, gcn=False, cuda=args.cuda)
        agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=args.cuda)
        enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, args.hidden, adj_lists, agg2,
                       base_model=enc1, gcn=False, cuda=args.cuda)
        enc1.num_samples = 5
        enc2.num_samples = 5

        self.enc = enc2
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, self.enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())


# In[42]:

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, Flag=0):
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


f1s, recs, precs, accs = [], [], [], []
for run_th in range(10):
    dataset_path = datasets[args.dataset]['dataset_path'][run_th]
    dataset_path = '/home/syf/workspace/jupyters/Nancy/H2GCN' / Path(dataset_path)
    dataset = utils.PlanetoidData(dataset_str=dataset_str, dataset_path=dataset_path, val_size=val_size)
    train_mask = dataset._dense_data['train_mask']
    test_mask = dataset._dense_data['test_mask']
    val_mask = dataset._dense_data['val_mask']
    per = int(round(0.025 * dataset.labels.shape[0] / dataset.label_count.shape[0]))
    val_lb = int(round(0.025 * dataset.labels.shape[0]))
    train_mask, val_mask, test_mask = random_planetoid_splits(dataset, dataset.label_count.shape[0], percls_trn=per,
                                                              val_lb=val_lb, Flag=0)
    train = list(np.where(train_mask)[0])
    test = np.where(test_mask)[0]
    val = np.where(val_mask)[0]

    graphsage = SupervisedGraphSage(feat_data, num_classes).to(device)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=args.lr)
    times = []
    val_f_list = []
    test_f_list = []
    for batch in range(args.epochs):
        batch_nodes = train[:256]
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes,
                              Variable(torch.LongTensor(labels[np.array(batch_nodes)])).to(device))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)

        val_f1 = f1_score(labels[val], graphsage.forward(val).data.cpu().numpy().argmax(axis=1), average="macro")
        test_f1 = f1_score(labels[test], graphsage.forward(test).data.cpu().numpy().argmax(axis=1), average="macro")
        val_f_list.append(val_f1)
        test_f_list.append(test_f1)
        if batch % 10 == 0:
            print("{}-th batch, train loss: {:.2}, Val F1: {:.2}, Test F1: {:.2}, train size: {}".format(
                batch, loss.data.item(), val_f1, test_f1, len(train)))

    x = [i for i in range(len(val_f_list))]
    plt.plot(x, val_f_list)
    plt.plot(x, test_f_list)
    plt.legend()
    plt.show()
    val_pred = graphsage.forward(val).data.cpu().numpy().argmax(axis=1)
    print("Validation F1:{:.2}, Rec:{:.2}, Prec:{:.2}, Acc:{:.2}".format(
        f1_score(labels[val], val_pred, average="macro"),
        recall_score(labels[val], val_pred, average="macro"),
        precision_score(labels[val], val_pred, average="macro"),
        accuracy_score(labels[val], val_pred)))

    test_pred = graphsage.forward(test).data.cpu().numpy().argmax(axis=1)
    f1, rec, prec, acc = (f1_score(labels[test], test_pred, average="macro"),
                          recall_score(labels[test], test_pred, average="macro"),
                          precision_score(labels[test], test_pred, average="macro"),
                          accuracy_score(labels[test], test_pred))
    print("Test F1:{:.2}, Rec:{:.2}, Prec:{:.2}, Acc:{:.2}".format(f1, rec, prec, acc))

    f1s.append(f1);
    recs.append(rec);
    precs.append(prec);
    accs.append(acc)
#     results.append(f1_score(labels[test], test_output.data.numpy().argmax(axis=1), average="micro"))


print("Average Test F1:{:.2}({:.2}), Rec:{:.2}({:.2}), Prec:{:.2}({:.2}), Acc:{:.2}({:.2})".format(
    np.mean(f1s), np.std(f1s), np.mean(recs), np.std(recs),
    np.mean(precs), np.std(precs), np.mean(accs), np.std(accs)))

# In[43]:


f1s2, recs2, precs2, accs2 = [], [], [], []
for run_th in range(10):
    dataset_path = datasets[args.dataset]['dataset_path'][run_th]
    dataset_path = '/home/syf/workspace/jupyters/Nancy/H2GCN' / Path(dataset_path)
    dataset = utils.PlanetoidData(dataset_str=dataset_str, dataset_path=dataset_path, val_size=val_size)
    train_mask = dataset._dense_data['train_mask']
    test_mask = dataset._dense_data['test_mask']
    val_mask = dataset._dense_data['val_mask']
    per = int(round(0.025 * dataset.labels.shape[0] / dataset.label_count.shape[0]))
    val_lb = int(round(0.025 * dataset.labels.shape[0]))
    train_mask, val_mask, test_mask = random_planetoid_splits(dataset, dataset.label_count.shape[0], percls_trn=per,
                                                              val_lb=val_lb, Flag=0)
    train = list(np.where(train_mask)[0])
    test = np.where(test_mask)[0]
    val = np.where(val_mask)[0]
    val_test = np.concatenate([val, test], 0)

    graphsage = SupervisedGraphSage(feat_data, num_classes).to(device)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=args.lr)

    times = []
    train_ori = train
    clamp_labels = np.array([labels[i] if i in train else [-1] for i in range(n_nodes)])
    for batch in range(500):
        batch_nodes = train[:256]
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes,
                              Variable(torch.LongTensor(clamp_labels[np.array(batch_nodes)])).to(device))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)

        val_test_output = F.softmax(graphsage.forward(val_test), dim=1)
        val_test_prob, val_test_pred = torch.max(val_test_output, 1)
        clamp_nodes = list(val_test[val_test_prob.cpu() > 0.9])
        train = train_ori + clamp_nodes

        all_pred = F.softmax(graphsage.forward(range(n_nodes)), dim=1).data.cpu().numpy().argmax(axis=1)
        for i in clamp_nodes:
            clamp_labels[i] = [all_pred[i]]
        random.shuffle(train)

        val_f1 = f1_score(labels[val], graphsage.forward(val).data.cpu().numpy().argmax(axis=1), average="macro")
        test_f1 = f1_score(labels[test], graphsage.forward(test).data.cpu().numpy().argmax(axis=1), average="macro")
        if batch % 10 == 0:
            print("{}-th batch, train loss: {:.2}, Val F1: {:.2}, Test F1: {:.2}, train size: {}".format(
                batch, loss.data.item(), val_f1, test_f1, len(train)))
    #         if batch % 10 == 0:
    #             print("{}-th batch, train loss: {:.2}, Val+Test F1: {:.2}, train size: {}".format(batch, loss.data.item(),
    #                   f1_score(labels[val_test], val_test_output.data.numpy().argmax(axis=1), average="macro"), len(train)))

    val_pred = graphsage.forward(val).data.cpu().numpy().argmax(axis=1)
    print("Validation F1:{:.2}, Rec:{:.2}, Prec:{:.2}, Acc:{:.2}".format(
        f1_score(labels[val], val_pred, average="macro"),
        recall_score(labels[val], val_pred, average="macro"),
        precision_score(labels[val], val_pred, average="macro"),
        accuracy_score(labels[val], val_pred)))

    test_pred = graphsage.forward(test).data.cpu().numpy().argmax(axis=1)
    f1, rec, prec, acc = (f1_score(labels[test], test_pred, average="macro"),
                          recall_score(labels[test], test_pred, average="macro"),
                          precision_score(labels[test], test_pred, average="macro"),
                          accuracy_score(labels[test], test_pred))
    print("Test F1:{:.2}, Rec:{:.2}, Prec:{:.2}, Acc:{:.2}".format(f1, rec, prec, acc))

    f1s2.append(f1);
    recs2.append(rec);
    precs2.append(prec);
    accs2.append(acc)
#     results.append(f1_score(labels[test], test_output.data.numpy().argmax(axis=1), average="micro"))


print("Average Test F1:{:.2}({:.2}), Rec:{:.2}({:.2}), Prec:{:.2}({:.2}), Acc:{:.2}({:.2})".format(
    np.mean(f1s2), np.std(f1s2), np.mean(recs2), np.std(recs2),
    np.mean(precs2), np.std(precs2), np.mean(accs2), np.mean(accs2)))

# In[44]:


print("GCN Average Test F1:{:.3}({:.3}), Rec:{:.3}({:.3}), Prec:{:.3}({:.3}), Acc:{:.3}({:.3})".format(
    np.mean(f1s), np.std(f1s), np.mean(recs), np.std(recs),
    np.mean(precs), np.std(precs), np.mean(accs), np.std(accs)))

# In[45]:


print("GCN (label enhance) Average Test F1:{:.3}({:.3}), Rec:{:.3}({:.3}), Prec:{:.3}({:.3}), Acc:{:.3}({:.3})".format(
    np.mean(f1s2), np.std(f1s2), np.mean(recs2), np.std(recs2),
    np.mean(precs2), np.std(precs2), np.mean(accs2), np.mean(accs2)))

# In[ ]:
