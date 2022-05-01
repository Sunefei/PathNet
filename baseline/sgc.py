#参考https://github.com/pyg-team/pytorch_geometric/blob/master/examples/sgc.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import SGConv

class SGC(torch.nn.Module):
    def __init__(self,input_dim,output_dim,K,cached=True,add_self_loops=True):
        super(SGC, self).__init__()
        self.conv1 = SGConv(input_dim,output_dim,K=K,cached=cached,
                            add_self_loops=add_self_loops)

    def forward(self,x,edge_index):
        x = self.conv1(x, edge_index)
        return {'out':F.log_softmax(x, dim=1),'emb':x}  #dim=-1就是dim=1