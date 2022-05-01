import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv

class GAT(nn.Module):
    """
    model_init_param需要num_layers,hidden_unit,dropout_rate,heads
    model_forward_param需要x,edge_index
    模型结构：num_layers（最少为2）-1层(卷积网络+BN)+卷积网络
    """
    def __init__(self,num_layers,hidden_unit,input_dim,output_dim,dropout_rate,heads):
        super(GAT,self).__init__()

        self.num_layers=num_layers
        self.dropout_rate=dropout_rate

        self.convs=nn.ModuleList()
        self.convs.append(GATConv(input_dim,hidden_unit,heads))
        for i in range(num_layers-2):
            self.convs.append(GATConv(hidden_unit*heads,hidden_unit,heads))
        self.convs.append(GATConv(hidden_unit*heads,output_dim,1,concat=False))

        self.bns=nn.ModuleList([nn.BatchNorm1d(hidden_unit*heads) for i in range(num_layers-1)])

    
    def forward(self,x,edge_index):
        for i in range(self.num_layers-1):
            x=self.convs[i](x,edge_index)
            x=self.bns[i](x)
        x=self.convs[self.num_layers-1](x,edge_index)
        
        return {'out':F.log_softmax(x, dim=1),'emb':x}