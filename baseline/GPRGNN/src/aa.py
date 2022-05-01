#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.autograd import Variable
from torch.nn import init

from aggregators import MeanAggregator, MeanStdAggregator, MeanStdSelfAggregator
from encoders import Encoder, STDEncoder
from utils import load_data, load_cora, rand_split, geom_split, load_citeseer, load_pubmed

# In[2]:


import argparse

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='normal', help='std or normal')
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

args = parser.parse_args(args=['--dataset', 'cora', '--no-cuda', '--hidden', '32', '--lr', '0.7'])
args.cuda = not args.no_cuda and torch.cuda.is_available()

print(args)

device = torch.device('cuda:0')


# In[3]:


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


# In[4]:


if args.dataset == "cora":
    feat_data, labels, adj_lists, n_nodes, n_feats, num_classes = load_cora()
elif args.dataset == "chameleon" or "texas" or "squirrel":
    feat_data, labels, adj_lists, n_nodes, n_feats, num_classes = load_data(dataset_str=args.dataset,
                                                                            directed_graph=False, self_loop=False)
elif args.dataset == "citeseer":
    feat_data, labels, adj_lists, n_nodes, n_feats, num_classes = load_citeseer()
elif args.dataset == "pubmed":
    feat_data, labels, adj_lists, n_nodes, n_feats, num_classes = load_pubmed()

# In[5]:

f1s, recs, precs, accs = [], [], [], []
for run_th in range(10):
    train, val, test = geom_split(args.dataset, run_th)
    val_test = np.concatenate([val, test], 0)

    graphsage = SupervisedGraphSage(feat_data, num_classes)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=args.lr)
    times = []
    for batch in range(args.epochs):
        batch_nodes = train[:256]
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes,
                              Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)

        val_test_output = F.softmax(graphsage.forward(val_test), dim=1)
        val_test_prob = torch.max(val_test_output, 1)[0]

        if batch % 10 == 0:
            print("{}-th batch, train loss: {:.2}, Val+Test F1: {:.2}, train size: {}".format(batch, loss.data.item(),
                                                                                              f1_score(labels[val_test],
                                                                                                       val_test_output.data.numpy().argmax(
                                                                                                           axis=1),
                                                                                                       average="micro"),
                                                                                              len(train)))

    val_pred = graphsage.forward(val).data.numpy().argmax(axis=1)
    print("Validation F1:{:.2}, Rec:{:.2}, Prec:{:.2}, Acc:{:.2}".format(
        f1_score(labels[val], val_pred, average="macro"),
        recall_score(labels[val], val_pred, average="macro"),
        precision_score(labels[val], val_pred, average="macro"),
        accuracy_score(labels[val], val_pred)))

    test_pred = graphsage.forward(test).data.numpy().argmax(axis=1)
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

# In[6]:


f1s2, recs2, precs2, accs2 = [], [], [], []
for run_th in range(10):
    train, val, test = geom_split(args.dataset, run_th)
    val_test = np.concatenate([val, test], 0)

    graphsage = SupervisedGraphSage(feat_data, num_classes)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)

    times = []
    train_ori = train
    clamp_labels = np.array([labels[i] if i in train else [-1] for i in range(n_nodes)])
    for batch in range(args.epochs):
        batch_nodes = train[:256]
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes,
                              Variable(torch.LongTensor(clamp_labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)

        val_test_output = F.softmax(graphsage.forward(val_test), dim=1)
        val_test_prob, val_test_pred = torch.max(val_test_output, 1)
        clamp_nodes = list(val_test[val_test_prob > 0.9])
        train = train_ori + clamp_nodes

        all_pred = F.softmax(graphsage.forward(range(n_nodes)), dim=1).data.numpy().argmax(axis=1)
        for i in clamp_nodes:
            clamp_labels[i] = [all_pred[i]]
        random.shuffle(train)

        if batch % 10 == 0:
            print("{}-th batch, train loss: {:.2}, Val+Test F1: {:.2}, train size: {}".format(batch, loss.data.item(),
                                                                                              f1_score(labels[val_test],
                                                                                                       val_test_output.data.numpy().argmax(
                                                                                                           axis=1),
                                                                                                       average="micro"),
                                                                                              len(train)))

    val_pred = graphsage.forward(val).data.numpy().argmax(axis=1)
    print("Validation F1:{:.2}, Rec:{:.2}, Prec:{:.2}, Acc:{:.2}".format(
        f1_score(labels[val], val_pred, average="macro"),
        recall_score(labels[val], val_pred, average="macro"),
        precision_score(labels[val], val_pred, average="macro"),
        accuracy_score(labels[val], val_pred)))

    test_pred = graphsage.forward(test).data.numpy().argmax(axis=1)
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

# In[7]:


print("GCN Average Test F1:{:.3}({:.3}), Rec:{:.3}({:.3}), Prec:{:.3}({:.3}), Acc:{:.3}({:.3})".format(
    np.mean(f1s), np.std(f1s), np.mean(recs), np.std(recs),
    np.mean(precs), np.std(precs), np.mean(accs), np.std(accs)))

# In[8]:


print("GCN (label enhance) Average Test F1:{:.3}({:.3}), Rec:{:.3}({:.3}), Prec:{:.3}({:.3}), Acc:{:.3}({:.3})".format(
    np.mean(f1s2), np.std(f1s2), np.mean(recs2), np.std(recs2),
    np.mean(precs2), np.std(precs2), np.mean(accs2), np.mean(accs2)))

# In[ ]:
