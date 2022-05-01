#参考自：https://github.com/pyg-team/pytorch_geometric/blob/74245f3a680c1f6fd1944623e47d9e677b43e827/benchmark/kernel/gin.py

import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv

class GIN(torch.nn.Module):
    def __init__(self,input_dim,output_dim,hidden_unit,num_layers,dropout_rate):
        super(GIN, self).__init__()

        self.dropout_rate=dropout_rate

        self.conv1 = GINConv(
            Sequential(
                Linear(input_dim,hidden_unit),
                ReLU(),
                Linear(hidden_unit, hidden_unit),
                ReLU(),
                BN(hidden_unit),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden_unit, hidden_unit),
                        ReLU(),
                        Linear(hidden_unit, hidden_unit),
                        ReLU(),
                        BN(hidden_unit),
                    ), train_eps=True))
        self.lin1 = Linear(hidden_unit, hidden_unit)
        self.lin2 = Linear(hidden_unit,output_dim)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self,x,edge_index):
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        return {'out':F.log_softmax(x, dim=-1),'emb':x}