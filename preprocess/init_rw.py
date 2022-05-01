import numpy as np
import json
import time
from pathlib import Path
from dataset import PlanetoidData
from torch_geometric.utils import from_scipy_sparse_matrix
import torch
import compute_merw as rw
import scipy
import argparse
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('-data', '--data_name',
                    # action='append', nargs='*',
                    type=str, default='cora')  # , 'citeseer']
args = parser.parse_args()
data_name = args.data_name


def load_data_ranked(name):
    datasets = json.load(
        open("/home/syf/workspace/jupyters/configs/dataset.json"))
    dataset_run = datasets[name]["dataset"]
    dataset_path = datasets[name]["dataset_path"][0]
    # dataset_path = "/home/syf/workspace/jupyters" / Path(dataset_path)
    val_size = datasets[name]["val_size"]

    dataset = PlanetoidData(
        dataset_str=dataset_run, dataset_path=dataset_path, val_size=val_size
    )

    # features = dataset._sparse_data["features"]
    adj = dataset._sparse_data["sparse_adj"]
    n = adj.shape[0]
    labels = dataset._dense_data["y_all"]
    # adj = adj + scipy.sparse.eye(n)

    edge_index = from_scipy_sparse_matrix(adj)[0]  # indices + edge_weight
    # x = np.array( features.todense() )
    edge_index = np.array(edge_index)
    y = torch.tensor(np.argmax(labels, 1), dtype=torch.long)
    return edge_index, adj, y


if __name__ == '__main__':
    old_datasets = ["cora", "pubmed", "citeseer", "cornell"]
    for data_name in [
        "cornell",
        "cora",
        # 'Nba',
        "citeseer",
        "pubmed",
        # 'Electronics',
        # 'bgp',
    ]:
        if data_name in old_datasets:
            edge_index, adj, y = load_data_ranked(data_name)
        else:
            y = np.load(f"/data/syf/{data_name}/y.npy")
            edge_index = np.load(f"/data/syf/{data_name}/edge_index.npy")
            row = edge_index[0]
            col = edge_index[1]
            data = np.ones(edge_index.shape[-1])
            adj = csr_matrix((data, (row, col)),
                             shape=(y.shape[0], y.shape[0]))
            n = y.shape[0]
            # adj = adj + scipy.sparse.eye(n)  # with self-loop or not
        start = time.time()
        start_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(start))
        print("calculating", start_time)
        # print(type(adj))
        P_merw, _, _, _ = rw.compute_merw(adj)

        M = edge_index.shape[1]

        cal_end = time.time()
        print("saving", (cal_end-start)/60, (cal_end-start)/3600)
        file = open(f"edge_input/{data_name}_nsl.in", "w")
        print(y.shape[0], edge_index.shape[1]*2, file=file)
        for i in range(M):
            u, v = edge_index[0, i], edge_index[1, i]
            print(u, v, P_merw[u, v], file=file)
            print(v, u, P_merw[v, u], file=file)
        end = time.time()
        end_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(end))
        print("over", (end-start)/60, (end-start)/3600, end_time)
