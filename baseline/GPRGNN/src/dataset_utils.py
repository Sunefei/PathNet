#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch
import ipdb
import math
import ipdb

import os.path as osp
import numpy as np
import torch.nn.functional as F
import torch_geometric.transforms as T

from cSBM_dataset import dataset_ContextualSBM
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import Amazon
from torch_geometric.nn import APPNP
from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import is_undirected, to_undirected
from torch_geometric.io import read_npz


class dataset_heterophily(InMemoryDataset):
    def __init__(self, root='data/', name=None,
                 p2raw=None,
                 train_percent=0.01,
                 transform=None, pre_transform=None):

        existing_dataset = ['chameleon', 'film', 'squirrel']
        if name not in existing_dataset:
            raise ValueError(
                f'name of hypergraph dataset must be one of: {existing_dataset}')
        else:
            self.name = name

        self._train_percent = train_percent

        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(
                f'path to raw hypergraph dataset "{p2raw}" does not exist!')

        if not osp.isdir(root):
            os.makedirs(root)

        self.root = root

        super(dataset_heterophily, self).__init__(
            root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_percent = self.data.train_percent.item()

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        p2f = osp.join(self.raw_dir, self.name)
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


class WebKB(InMemoryDataset):
    r"""The WebKB datasets used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features are the bag-of-words representation of web pages.
    The task is to classify the nodes into one of the five categories, student,
    project, course, staff, and faculty.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cornell"`,
            :obj:`"Texas"` :obj:`"Washington"`, :obj:`"Wisconsin"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/'
           'master/new_data')

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['cornell', 'texas', 'washington', 'wisconsin']

        super(WebKB, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index = to_undirected(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


def DataLoader(name):
    # assert name in ['cSBM_data_Aug_19_2020-13:06',
    #                 'cSBM_data_Aug_18_2020-18:50',
    #                 'cSBM_data_Aug_21_2020-10:06',
    #                 'cSBM_data_Aug_19_2020-20:41',
    #                 'cSBM_data_Aug_21_2020-11:04',
    #                 'cSBM_data_Aug_21_2020-11:21',
    #                 'cSBM_data_Sep_01_2020-14:15',
    #                 'cSBM_data_Sep_01_2020-14:18',
    #                 'cSBM_data_Sep_01_2020-14:19',
    #                 'cSBM_data_Sep_01_2020-14:32',
    #                 'cSBM_data_Sep_01_2020-14:22',
    #                 'cSBM_data_Sep_01_2020-14:23',
    #                 'cSBM_data_Sep_01_2020-14:27',
    #                 'cSBM_data_Sep_01_2020-14:29',
    #                 'Cora', 'Citeseer', 'PubMed',
    #                 'Computers', 'Photo',
    #                 'chameleon', 'film', 'squirrel',
    #                 'Texas', 'Cornell']

    # if name in ['cSBM_data_Aug_19_2020-13:06',
    #             'cSBM_data_Aug_18_2020-18:50',
    #             'cSBM_data_Aug_21_2020-10:06',
    #             'cSBM_data_Aug_19_2020-20:41',
    #             'cSBM_data_Aug_21_2020-11:04',
    #             'cSBM_data_Aug_21_2020-11:21',
    #             'cSBM_data_Sep_01_2020-14:15',
    #             'cSBM_data_Sep_01_2020-14:18',
    #             'cSBM_data_Sep_01_2020-14:19',
    #             'cSBM_data_Sep_01_2020-14:32',
    #             'cSBM_data_Sep_01_2020-14:22',
    #             'cSBM_data_Sep_01_2020-14:23',
    #             'cSBM_data_Sep_01_2020-14:27',
    #             'cSBM_data_Sep_01_2020-14:29']:
    if 'cSBM_data' in name:
        path = './data/'
        dataset = dataset_ContextualSBM(path, name=name)
    print(name)
    if name in ['cora', 'citeseer', 'pubmed']:
        # root_path = '/home/srtpgroup/baseline_other/src/'
        root_path = './'
        # path = osp.join(root_path, 'data/cora/', name)
        path = './data/pubmed/'
        print(path)
        # dataset = Planetoid(path, name, transform=T.NormalizeFeatures())
        dataset = Planetoid(
            root=path, name=name.capitalize()
        )
        # dataset = Planetoid(path, name)
    elif name in ['computers', 'photo']:
        root_path = './'
        path = osp.join(root_path, 'data', name)
        dataset = Amazon(path, name, T.NormalizeFeatures())
    elif name in ['chameleon', 'film', 'squirrel']:
        path = '../data/squirrel/'
        print(path)
        dataset = dataset_heterophily(
            root='../data/', name=name, transform=T.NormalizeFeatures())
    elif name in ['texas', 'cornell']:
        path = './data/texas/'
        print(path)
        dataset = WebKB(root='./data/',
                        name=name, transform=T.NormalizeFeatures())
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')

    return dataset
