'''
Some components of this code are adapted from https://github.com/tkipf/gcn

@inproceedings{kipf2017semi,
    title={Semi-Supervised Classification with Graph Convolutional Networks},
    author={Kipf, Thomas N. and Welling, Max},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2017}
}

and https://github.com/graphdml-uiuc-jlu/geom-gcn/ (with bugs fixed)

Modified by Jiong Zhu (jiongzhu@umich.edu)
'''

# %%
import warnings
from enum import Enum
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import multiprocessing
from itertools import chain
from argparse import Namespace
from pathlib import Path

# %%


class TransformAdj:
    @staticmethod
    def nhood(adj: np.ndarray, nhood, keep_diag=False):
        assert adj.ndim == 2 and adj.shape[0] == adj.shape[1]
        if np.isnan(nhood):
            return np.ones(adj.shape)

        mt = np.eye(adj.shape[1])
        i = 0
        edge_sum = 0
        while i < nhood:
            mt = mt @ (adj + np.eye(adj.shape[0]))
            mt = (mt > 0).astype(mt.dtype)
            new_edge_sum = np.sum(mt)
            if edge_sum == new_edge_sum:
                break
            else:
                edge_sum = new_edge_sum
            i += 1

        if keep_diag:
            diag_ind = np.diag_indices_from(adj)
            mt[diag_ind] = adj[diag_ind]

        return mt

    @staticmethod
    def nhoodSplit(adj: np.ndarray, nhood):
        assert adj.ndim == 2 and adj.shape[0] == adj.shape[1]
        if np.isnan(nhood):
            return np.ones(adj.shape)
        mt = np.eye(adj.shape[1])
        mtList = [mt]
        i = 0
        edge_sum = 0
        while i < nhood:
            prev_mt = mt
            # TODO: increase performance for the following line on large graph like citeseer
            mt = mt @ (adj + np.eye(adj.shape[0]))
            mt = (mt > 0).astype(mt.dtype)
            new_edge_sum = np.sum(mt)
            if edge_sum == new_edge_sum:
                break
            else:
                edge_sum = new_edge_sum
            i += 1
            mtList.append(mt - prev_mt)
        return mtList

    class LType(Enum):
        ORDINARY = 0
        SYM_NORMALIZED = 1
        RW_NORMALIZED = 2

    @classmethod
    def laplacian(cls, adj: np.ndarray, Ltype):
        assert type(adj) is np.ndarray
        D = np.diag(np.sum(adj, axis=1))
        L = D - adj
        if Ltype == cls.LType.ORDINARY:
            return L
        elif Ltype == cls.LType.SYM_NORMALIZED:
            L = np.sqrt(D) @ L @ np.sqrt(D)
            return L
        elif Ltype == cls.LType.RW_NORMALIZED:
            raise NotImplementedError()
        else:
            raise ValueError(
                f"Parameter Ltype must be selected from {cls.LType}.")


class TransformSPAdj:
    class NType(Enum):
        ORDINARY = 0
        SYM_NORMALIZED = 1
        RW_NORMALIZED = 2
        CHEBY = 3

    @classmethod
    def normalize(cls, adj, Ntype):
        if Ntype == cls.NType.ORDINARY:
            return adj
        else:
            if Ntype == cls.NType.SYM_NORMALIZED:
                degInvSqrt = np.power(adj.sum(axis=1).A1, -0.5)
                degInvSqrt[np.isinf(degInvSqrt)] = 0.
                DInvSqrt = sp.diags(degInvSqrt)
                adjNormalized = DInvSqrt @ adj @ DInvSqrt
            elif Ntype == cls.NType.RW_NORMALIZED:
                degInv = np.power(adj.sum(axis=1).A1, -1)
                degInv[np.isinf(degInv)] = 0.
                DInv = sp.diags(degInv)
                adjNormalized = DInv @ adj
            return adjNormalized

    @staticmethod
    def addEye(adj: sp.csr_matrix):
        adj = adj.tolil(copy=True)
        adj.setdiag(1)
        return adj.tocsr()

    @staticmethod
    def removeEye(adj: sp.csr_matrix):
        adj = adj.tolil(copy=True)
        adj.setdiag(0)
        return adj.tocsr()

    @classmethod
    def nhoodSplit(cls, adj: sp.csr_matrix, nhood):
        assert adj.ndim == 2 and adj.shape[0] == adj.shape[1]
        if np.isnan(nhood):
            return sp.csr_matrix(np.ones(adj.shape))
        mt = sp.eye(adj.shape[1])
        mtList = [mt]
        i = 0
        edge_sum = 0
        while i < nhood:
            prev_mt = mt
            mt = mt @ (adj + sp.eye(adj.shape[0]))
            mt = (mt > 0).astype(mt.dtype)
            new_edge_sum = mt.sum()
            if edge_sum == new_edge_sum:
                break
            else:
                edge_sum = new_edge_sum
            i += 1
            mtList.append(mt - prev_mt)
        return mtList


class PlanetoidData:
    @staticmethod
    def parse_index_file(filename):
        """Parse index file."""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    @staticmethod
    def sample_mask(idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)

    @staticmethod
    def _pkl_load(f):
        if sys.version_info > (3, 0):
            return pkl.load(f, encoding='latin1')
        else:
            return pkl.load(f)

    @staticmethod
    def graphDict2Adj(graph):
        return nx.adjacency_matrix(nx.from_dict_of_lists(graph), nodelist=range(len(graph)))

    def getNXGraph(self):
        G = nx.from_scipy_sparse_matrix(self.sparse_adj)
        for i, label in enumerate(self.labels):
            # To match the synthetic graph, label begins from 1.
            G.nodes[i]["color"] = int(label + 1)
        return G

    def load_data(self, dataset_str, dataset_path="data", save_plot=None, val_size=None):
        """
        Loads input data from gcn/data directory

        ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
            (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
        ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
        ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
            object;
        ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

        All objects above must be saved using python pickle module.

        :param dataset_str: Dataset name
        :return: All data input files loaded (as well the training/test data).
        """
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("{}/{}.{}".format(dataset_path, dataset_str, names[i]), 'rb') as f:
                objects.append(self._pkl_load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = self.parse_index_file(
            "{}/{}.test.index".format(dataset_path, dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        if len(test_idx_range_full) != len(test_idx_range):
            print(
                f"Patch for citeseer dataset is applied for dataset {dataset_str} at {dataset_path}")
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended
            self.non_valid_samples = set(
                test_idx_range_full) - set(test_idx_range)
        else:
            self.non_valid_samples = set()

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = self.graphDict2Adj(graph).astype(np.float32)

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        # Fix citeseer (and GeomGCN) bug
        self.non_valid_samples = self.non_valid_samples.union(
            set(list(np.where(labels.sum(1) == 0)[0])))

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))

        train_mask = self.sample_mask(idx_train, labels.shape[0])
        test_mask = self.sample_mask(idx_test, labels.shape[0])
        val_mask = np.bitwise_not(np.bitwise_or(train_mask, test_mask))
        if val_size is not None:
            if np.sum(val_mask) > val_size:
                idx_val = range(len(y), len(y) + val_size)
                val_mask = self.sample_mask(idx_val, labels.shape[0])
            else:
                print(
                    f"Val set size set to {np.sum(val_mask)} due to insufficient samples.")
        wild_mask = np.bitwise_not(train_mask + val_mask + test_mask)

        for n_i in self.non_valid_samples:
            if train_mask[n_i]:
                warnings.warn("Non valid samples detected in training set")
                train_mask[n_i] = False
            elif test_mask[n_i]:
                warnings.warn("Non valid samples detected in test set")
                test_mask[n_i] = False
            elif val_mask[n_i]:
                warnings.warn("Non valid samples detected in val set")
                val_mask[n_i] = False
            wild_mask[n_i] = False

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_wild = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]
        y_wild[wild_mask, :] = labels[wild_mask, :]

        self._sparse_data["sparse_adj"] = adj
        self._sparse_data["features"] = features
        self._dense_data["y_all"] = labels
        self._dense_data["train_mask"] = train_mask
        self._dense_data["val_mask"] = val_mask
        self._dense_data["test_mask"] = test_mask
        self._dense_data["y_train"] = y_train
        self._dense_data["y_val"] = y_val
        self._dense_data["y_test"] = y_test
        self._dense_data["wild_mask"] = wild_mask
        self._dense_data["y_wild"] = y_wild
        self.__preprocessedAdj = None
        self.__preprocessedFeature = None

        return adj, features, y_train, y_val, y_test, y_wild, train_mask, val_mask, test_mask, wild_mask, labels

    def __getattribute__(self, name):
        if name in ("_sparse_data", "_dense_data"):
            return object.__getattribute__(self, name)
        elif name in self._sparse_data:
            return self._sparse_data[name]
        elif name in self._dense_data:
            return self._dense_data[name]
        else:
            return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if name in ("_sparse_data", "_dense_data"):
            object.__setattr__(self, name, value)
        elif name in self._sparse_data:
            self._sparse_data[name] = value
        elif name in self._dense_data:
            self._dense_data[name] = value
        else:
            object.__setattr__(self, name, value)

    def __init__(self, dataset_str, dataset_path, val_size=None):
        self._sparse_data = dict()
        self._dense_data = dict()
        self.dataset_str = dataset_str
        self.dataset_path = dataset_path
        self.load_data(dataset_str, dataset_path, val_size=val_size)
        self._original_data = (self._sparse_data.copy(),
                               self._dense_data.copy())

    def reload_data(self):
        self._sparse_data, self._dense_data = self._original_data
        self.__preprocessedAdj = None
        self.__preprocessedFeature = None

    @property
    def labels(self):
        idx, labels = np.where(self.y_all)
        labels = labels.astype(np.int32)
        if len(idx) != self.num_samples:  # Citeseer bug
            part_labels = labels
            labels = np.zeros(self.num_samples) - 1
            labels[idx] = part_labels
        assert len(labels) == self.num_samples
        return labels

    @property
    def load_data_result(self):
        return (self.sparse_adj, self.features, self.y_train, self.y_val, self.y_test, self.train_mask, self.val_mask, self.test_mask)

    @property
    def num_labels(self):
        return self.y_all.shape[1]

    @property
    def num_samples(self):
        return self.features.shape[0]

    @property
    def train_features(self):
        return self.features[self.train_mask]

    @property
    def val_features(self):
        return self.features[self.val_mask]

    @property
    def test_features(self):
        return self.features[self.test_mask]

    @property
    def feature_dim(self):
        return self.features.shape[1]

    def get_sample_mask(self, label=slice(None), *scopes):
        if len(scopes) == 0:
            scope = ["train", "val", "test"]
        if type(label) is not slice:
            label = np.array(label).reshape(-1)
        label_mask = np.zeros(self.features.shape[0], dtype=bool)
        for scope in scopes:
            if scope == "train":
                label_mask_new = np.any(self.y_train[:, label] == 1, axis=1)
            elif scope == "val":
                label_mask_new = np.any(self.y_val[:, label] == 1, axis=1)
            elif scope == "test":
                label_mask_new = np.any(self.y_test[:, label] == 1, axis=1)
            elif scope == "wild":
                label_mask_new = np.any(self.y_wild[:, label] == 1, axis=1)
            else:
                raise ValueError("Unknown scope {}".format(scope))
            label_mask = np.bitwise_or(label_mask, label_mask_new)
        return label_mask

    def feature_iter(self, label=slice(None), *scope):
        if len(scope) == 0:
            scope = ["train", "val", "test"]

        if len(scope) > 1:
            for group in scope:
                for feature in self.feature_iter(label, group):
                    yield feature
        else:
            scope = scope[0]
            if type(label) is not slice:
                label = np.array(label).reshape(-1)

            if scope == "train":
                label_mask = np.any(self.y_train[:, label] == 1, axis=1)
            elif scope == "val":
                label_mask = np.any(self.y_val[:, label] == 1, axis=1)
            elif scope == "test":
                label_mask = np.any(self.y_test[:, label] == 1, axis=1)
            elif scope == "wild":
                label_mask = np.any(self.y_wild[:, label] == 1, axis=1)
            else:
                raise ValueError("Unknown scope {}".format(scope))
            feature_list = self.features[label_mask, :]
            for i in range(feature_list.shape[0]):
                yield feature_list[i]

    def feature_label_iter(self, *scope):
        '''
        Test code:
        cora = utils.PlanetoidData("ind.cora", "../../baselines/GAT/data")
        for feature, i in cora.feature_label_iter():
            print(feature, i)
        '''
        for label in range(self.num_labels):
            for feature in self.feature_iter(label, *scope):
                yield feature, label

    def sort_label_by_size(self, descending=True):
        if descending:
            return np.argsort(self.label_count)[::-1]
        else:
            return np.argsort(self.label_count)

    @property
    def label_count(self, *scope):
        if len(scope) == 0:
            scope = ["train", "val", "test"]

        result = np.zeros(self.y_train.shape[1])
        for group in scope:
            if group == "train":
                result += np.sum(self.y_train, axis=0)
            elif group == "val":
                result += np.sum(self.y_val, axis=0)
            elif group == "test":
                result += np.sum(self.y_test, axis=0)
            elif group == "wild":
                result += np.sum(self.y_wild, axis=0)
            else:
                raise ValueError("Unknown scope {}".format(group))
        return result

    def feature_sample_eligible(self, label_count):
        if len(label_count) <= len(self.label_count):
            return np.all(np.sort(label_count)[::-1] <= np.sort(self.label_count)[::-1][:len(label_count)])
        else:
            return False

    def split_training_set(self, splits=2):
        self.train_mask_splits = np.zeros(
            (splits,) + self.train_mask.shape, dtype=self.train_mask.dtype)
        self.y_train_splits = np.zeros(
            (splits,) + self.y_train.shape, dtype=self.y_train.dtype)
        for label in range(self.y_train.shape[1]):
            avaliable_sample = np.where(self.y_train[:, label])[0]
            for i, index in enumerate(avaliable_sample):
                self.train_mask_splits[i %
                                       splits, index] = self.train_mask[index]
                self.y_train_splits[i %
                                    splits, index, :] = self.y_train[index, :]

    @property
    def preprocessedAdj(self):
        return self.__preprocessedAdj

    @property
    def preprocessedFeature(self):
        return self.__preprocessedFeature

    @property
    def __SPAdjTransform(self):
        self.__preprocessedAdj = True
        return TransformSPAdj

    @property
    def __NType(self):
        return TransformSPAdj.NType

    def adj_add_eye(self):
        self.sparse_adj = self.__SPAdjTransform.addEye(self.sparse_adj)

    def adj_remove_eye(self):
        self.sparse_adj = self.__SPAdjTransform.removeEye(self.sparse_adj)

    def get_eye(self):
        return sp.identity(self.num_samples, dtype=self.sparse_adj.dtype)

    def row_normalize_features(self):
        prev_err = np.seterr(divide='warn')
        rowSumInv = np.power(self.features.sum(1).A1, -1)
        rowSumInv[np.isinf(rowSumInv)] = 0.
        np.seterr(**prev_err)
        RSumInv = sp.diags(rowSumInv)
        self.features = RSumInv @ self.features
        self.__preprocessedFeature = True

    def preprocessGCN(self, add_eye=True):
        if self.preprocessedAdj == "GCN" and self.preprocessedFeature == "GCN":
            return
        elif self.preprocessedAdj or self.preprocessedFeature:
            self.reload_data()
        self._sparse_adj_raw = self.sparse_adj
        self._features_raw = self.features
        if add_eye > 0:
            self.adj_add_eye()
        elif add_eye < 0:
            self.adj_remove_eye()
        self.sparse_adj = self.__SPAdjTransform.normalize(
            self.sparse_adj, self.__NType.SYM_NORMALIZED)
        self.row_normalize_features()
        self.__preprocessedAdj = "GCN"
        self.__preprocessedFeature = "GCN"

    @classmethod
    def sparse2Tensor(cls, spmat, dtype=np.float32):
        import tensorflow as tf
        if type(spmat) is list:
            return [cls.sparse2Tensor(x) for x in spmat]
        X = spmat.tocoo()  # type: sp.coo_matrix
        indices = np.array([X.row, X.col]).T
        return tf.sparse.reorder(tf.SparseTensor(indices, X.data.astype(dtype), X.shape))

    def getTensors(self, getDenseAdj=False, getAdjHops=None, getAdjNormHops=None,
                   normType=TransformSPAdj.NType.SYM_NORMALIZED, dtype=np.float32):
        import tensorflow as tf
        tensors = Namespace()

        # Sparse tensor
        for key, value in self._sparse_data.items():
            setattr(tensors, key, self.sparse2Tensor(value, dtype))

        if getDenseAdj:
            tensors.adj = tf.sparse.to_dense(tensors.sparse_adj)
        else:
            tensors.adj = self.sparse2Tensor(self.sparse_adj, dtype)
        if getAdjHops:
            getAdjHops = [[int(x) for x in elem.split(",")]
                          for elem in getAdjHops]
            getAdjHopsMax = max(chain(*getAdjHops))
            adjSplits = TransformSPAdj.nhoodSplit(
                self.sparse_adj, getAdjHopsMax)
            # TODO: Consider using sparse tensor here - problem is scipy and tf does not support sparse tensor.
            adjhopsMerged = np.stack(
                [sum([adjSplits[i] for i in elem]).toarray() for elem in getAdjHops], axis=1)
            tensors.adj_hops = tf.constant(adjhopsMerged)
        if getAdjNormHops:
            getAdjNormHops = [[int(x) for x in elem.split(",")]
                              for elem in getAdjNormHops]
            getAdjHopsMax = max(chain(*getAdjNormHops))
            if normType == TransformSPAdj.NType.CHEBY:
                adjSplits = chebyshev_polynomials(
                    self.sparse_adj, getAdjHopsMax, eigenvalue=2, asspmat=True)
                adjSplitsNormed = [sum([adjSplits[i] for i in elem])
                            for elem in getAdjNormHops]
            else:
                adjSplits = TransformSPAdj.nhoodSplit(
                    self.sparse_adj, getAdjHopsMax)
                adjhopsSum = [sum([adjSplits[i] for i in elem])
                            for elem in getAdjNormHops]
                adjSplitsNormed = [TransformSPAdj.normalize(
                    x, normType) for x in adjhopsSum]
            tensors.adj_hops = [self.sparse2Tensor(
                x, dtype) for x in adjSplitsNormed]

        for key, value in self._dense_data.items():
            setattr(tensors, key, tf.constant(value, dtype=dtype))
        tensors.labels = tf.constant(self.labels)

        for name in ["preprocessedAdj", "preprocessedFeature"]:
            setattr(tensors, name, getattr(self, name))
        return tensors

    @property
    def dos_graph(self):
        sparse_adj_lil = self.sparse_adj.tolil()
        return {i: set(l) for i, l in enumerate(sparse_adj_lil.rows)}


class GeomGCNData(PlanetoidData):
    def __init__(self, dataset_str, dataset_path, splits_file_path=None, directed_graph=False,
                 adj_filename='out1_graph_edges.txt', feature_filename='out1_node_feature_label.txt'):
        self.load_data(dataset_str, dataset_path, splits_file_path, directed_graph,
                       adj_filename, feature_filename)

    def load_data(self, dataset_str, dataset_path, splits_file_path=None, directed_graph=False,
                  adj_filename='out1_graph_edges.txt', feature_filename='out1_node_feature_label.txt'):
        graph_adjacency_list_file_path = str(Path(dataset_path) / adj_filename)
        graph_node_features_and_labels_file_path = str(
            Path(dataset_path) / feature_filename)

        if directed_graph: # Original Geom-GCN use this
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset_str == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(
                        line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    # Fix uint8 to uint16 for the following line, adapted from Yujun to prevent overflow
                    feature_blank[np.array(
                        line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(
                        line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(
                        line[1].split(','), dtype=np.uint8)  # uint8 okay here since feature is 1-hot
                    graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes())).astype(np.float32)
        features = np.array(
            [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])]).astype(np.float32)
        labels = np.array(
            [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])]).astype(np.int32)
        y_all = np.zeros((labels.shape[0], labels.max() + 1))
        y_all[np.arange(y_all.shape[0]), labels] = 1

        self.sparse_adj = adj
        self.features = features
        self.y_all = y_all
        self.__preprocessedAdj = None
        self.__preprocessedFeature = None

        if splits_file_path:
            self.load_splits(splits_file_path)
        else:
            self.train_mask = None
            self.val_mask = None
            self.test_mask = None
            self.wild_mask = None
            self.splitted = False

    def load_splits(self, splits_file_path):
        raise NotImplementedError("wild_mask code is not correct.")
        with np.load(splits_file_path) as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']
            wild_mask = np.bitwise_not(train_mask + val_mask + test_mask)
        y_train = np.zeros(self.y_all.shape)
        y_val = np.zeros(self.y_all.shape)
        y_test = np.zeros(self.y_all.shape)
        y_wild = np.zeros(self.y_all.shape)
        y_train[train_mask, :] = self.y_all[train_mask, :]
        y_val[val_mask, :] = self.y_all[val_mask, :]
        y_test[test_mask, :] = self.y_all[test_mask, :]
        y_wild[wild_mask, :] = self.y_all[wild_mask, :]

        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.wild_mask = wild_mask
        self.y_wild = y_wild
        self.splitted = True

    @property
    def label_count(self, *scope):
        if not self.splitted:
            assert len(scope) == 0
            return self.y_all.sum(0)
        else:
            return super().label_count(*scope)


__data_cache = dict()
__data_cache_lock = multiprocessing.Lock()


def get_cora(val_size=None):
    if ("cora", val_size) not in __data_cache:
        __data_cache_lock.acquire()
        if ("cora", val_size) not in __data_cache:
            __data_cache[("cora", val_size)] = PlanetoidData(
                "ind.cora", "../../baselines/mixhop/data/planetoid/data/", val_size=val_size)
        __data_cache_lock.release()
    return __data_cache[("cora", val_size)]


def get_citeseer(val_size=None):
    if ("citeseer", val_size) not in __data_cache:
        __data_cache_lock.acquire()
        if ("citeseer", val_size) not in __data_cache:
            __data_cache[("citeseer", val_size)] = PlanetoidData(
                "ind.citeseer", "../../../baselines/mixhop/data/planetoid/data/", val_size=val_size)
        __data_cache_lock.release()
    return __data_cache[("citeseer", val_size)]


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i]
                      for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k, eigenvalue=None, asspmat=False):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    if eigenvalue is None:
        largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    else:
        largest_eigval = [eigenvalue]
    scaled_laplacian = (
        2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))
    
    if asspmat:
        return t_k
    else:
        return sparse_to_tuple(t_k)


# Test code for class
if __name__ == "__main__":
    cora = get_cora()
    cora.preprocessGCN()
    cora.getTensors()
    chameleon = GeomGCNData(
        "chameleon", "../../../baselines/geom-gcn/new_data/chameleon")
    film = GeomGCNData("film", "../../../baselines/geom-gcn/new_data/film")
