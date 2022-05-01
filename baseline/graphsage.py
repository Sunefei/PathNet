import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops

class GraphSAGE(nn.Module):
    """
    model_init_param需要num_layers,hidden_unit,dropout_rate
    model_forward_param需要x,edge_index
    模型结构：num_layers（最少为2）-1层(卷积网络+BN)+卷积网络
    """
    def __init__(self,num_layers,hidden_unit,input_dim,output_dim,dropout_rate):
        super(GraphSAGE,self).__init__()

        self.num_layers=num_layers
        self.dropout_rate=dropout_rate

        self.convs=nn.ModuleList()
        self.convs.append(my_SAGEConv(input_dim,hidden_unit))
        for i in range(num_layers-2):
            self.convs.append(my_SAGEConv(hidden_unit,hidden_unit))
        self.convs.append(my_SAGEConv(hidden_unit,output_dim))

        self.bns=nn.ModuleList([nn.BatchNorm1d(hidden_unit) for i in range(num_layers-1)])
    
    def forward(self,x,edge_index):
        for i in range(self.num_layers-1):
            x=self.convs[i](x,edge_index)
            x=self.bns[i](x)
        x=self.convs[self.num_layers-1](x,edge_index)
        
        return {'out':F.log_softmax(x, dim=1),'emb':x}



class my_SAGEConv(MessagePassing):

    def __init__(self, in_channels,out_channels,add_self_loops:bool = True,**kwargs):
        kwargs.setdefault('aggr', 'mean')
        super(my_SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops

        self.lin_l=nn.Linear(in_channels,out_channels)
        self.lin_r=nn.Linear(in_channels,out_channels)
    
    def forward(self, x, edge_index):
        x_l = self.lin_l(x)
        x_r = self.lin_r(x)

        if self.add_self_loops:
            num_nodes = x_l.size(0)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
        out = self.propagate(edge_index, x=x_l)

        out+=x_r
        
        return out
