import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import FAConv

class FAGCN(nn.Module):
    def __init__(self,num_layers,input_dim,hidden_unit,output_dim,dropout_rate,epsilon):
        super(FAGCN, self).__init__()
        self.eps = epsilon
        self.layer_num = num_layers
        self.dropout = dropout_rate

        self.layers = nn.ModuleList()
        for _ in range(self.layer_num):
            self.layers.append(FAConv(hidden_unit,epsilon,dropout_rate))

        self.t1 = nn.Linear(input_dim, hidden_unit)
        self.t2 = nn.Linear(hidden_unit, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self,x,edge_index):
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h,raw,edge_index)
        h = self.t2(h)
        return {'out':F.log_softmax(h, dim=1),'emb':h}
#参考自：https://github.com/bdy9527/FAGCN/blob/main/src/model.py