import numpy as np
import torch
from torch_scatter import scatter_add
from torch_geometric.utils import remove_self_loops


def edge_homophily(A, labels, ignore_negative=False):
    """ gives edge homophily, i.e. proportion of edges that are intra-class
    compute homophily of classes in labels vector
    See Zhu et al. 2020 "Beyond Homophily ..."
    if ignore_negative = True, then only compute for edges where nodes both have
        nonnegative class labels (negative class labels are treated as missing
    """
    src_node, targ_node = A.nonzero()
    matching = labels[src_node] == labels[targ_node]
    labeled_mask = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
    if ignore_negative:
        edge_hom = np.mean(matching[labeled_mask])
    else:
        edge_hom = np.mean(matching)
    return edge_hom

def compat_matrix(A, labels):
    """ c x c compatibility matrix, where c is number of classes
     H[i,j] is proportion of endpoints that are class j 
     of edges incident to class i nodes 
     See Zhu et al. 2020
    """
    c = len(np.unique(labels))
    H = np.zeros((c,c))
    src_node, targ_node = A.nonzero()
    for i in range(len(src_node)):
        src_label = labels[src_node[i]]
        targ_label = labels[targ_node[i]]
        H[src_label, targ_label] += 1
    H = H / np.sum(H, axis=1, keepdims=True)
    return H

def node_homophily(A, labels):
    """ average of homophily for each node
    """
    src_node, targ_node = A.nonzero()
    edge_idx = torch.tensor(np.vstack((src_node, targ_node)), dtype=torch.long).contiguous()
    labels = torch.tensor(labels)
    num_nodes = A.shape[0]
    return node_homophily_edge_idx(edge_idx, labels, num_nodes)

def edge_homophily_edge_idx(edge_idx, labels):
    """ edge_idx is 2x(number edges) """
    edge_index = remove_self_loops(edge_idx)[0]
    return torch.mean((labels[edge_index[0,:]] == labels[edge_index[1,:]]).float())

def node_homophily_edge_idx(edge_idx, labels, num_nodes):
    """ edge_idx is 2 x(number edges) """
    edge_index = remove_self_loops(edge_idx)[0]
    hs = torch.zeros(num_nodes)
    degs = torch.bincount(edge_index[0,:]).float()
    matches = (labels[edge_index[0,:]] == labels[edge_index[1,:]]).float()
    hs = hs.scatter_add(0, edge_index[0,:], matches) / degs
    return hs[degs != 0].mean()
    
def compat_matrix_edge_idx(edge_idx, labels):
    """
     c x c compatibility matrix, where c is number of classes
     H[i,j] is proportion of endpoints that are class j 
     of edges incident to class i nodes 
     "Generalizing GNNs Beyond Homophily"
     treats negative labels as unlabeled
     """
    edge_index = remove_self_loops(edge_idx)[0]
    src_node, targ_node = edge_index[0,:], edge_index[1,:]
    labeled_nodes = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
    label = labels.squeeze()
    c = label.max()+1
    H = torch.zeros((c,c)).to(edge_index.device)
    src_label = label[src_node[labeled_nodes]]
    targ_label = label[targ_node[labeled_nodes]]
    label_idx = torch.cat((src_label.unsqueeze(0), targ_label.unsqueeze(0)), axis=0)
    for k in range(c):
        sum_idx = torch.where(src_label == k)[0]
        add_idx = targ_label[sum_idx]
        scatter_add(torch.ones_like(add_idx).to(H.dtype), add_idx, out=H[k,:], dim=-1)
    H = H / torch.sum(H, axis=1, keepdims=True)
    return H

def our_measure(edge_index, label):
    """ 
    our measure \hat{h}
    treats negative labels as unlabeled 
    """
    label = label.squeeze()
    c = label.max()+1
    H = compat_matrix_edge_idx(edge_index, label)
    nonzero_label = label[label >= 0]
    counts = nonzero_label.unique(return_counts=True)[1]
    proportions = counts.float() / nonzero_label.shape[0]
    val = 0
    for k in range(c):
        class_add = torch.clamp(H[k,k] - proportions[k], min=0)
        if not torch.isnan(class_add):
            # only add if not nan
            val += class_add
    val /= c-1
    return val

