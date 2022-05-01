import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv

class GCN(nn.Module):
    """
    model_init_param需要num_layers,hidden_unit,dropout_rate
    model_forward_param需要x,edge_index
    模型结构：num_layers（最少为2）-1层(卷积网络+BN)+卷积网络
    """
    def __init__(self,num_layers,hidden_unit,input_dim,output_dim,dropout_rate):
        super(GCN,self).__init__()

        self.num_layers=num_layers
        self.dropout_rate=dropout_rate

        self.convs=nn.ModuleList()
        self.convs.append(GCNConv(input_dim,hidden_unit))
        for i in range(num_layers-2):
            self.convs.append(GCNConv(hidden_unit,hidden_unit))
        self.convs.append(GCNConv(hidden_unit,output_dim))

        self.bns=nn.ModuleList([nn.BatchNorm1d(hidden_unit) for i in range(num_layers-1)])
    
    def forward(self,x,edge_index):
        for i in range(self.num_layers-1):
            x=self.convs[i](x,edge_index)
            x=self.bns[i](x)
        x=self.convs[self.num_layers-1](x,edge_index)
        
        return {'out':F.log_softmax(x, dim=1),'emb':x}