import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    model_init_param需要num_layers,hidden_unit,dropout_rate
    model_forward_param需要x
    模型结构：num_layers（至少为2）-1层(线性网络+BN) + 线性网络
    """
    def __init__(self,num_layers,hidden_unit,input_dim,output_dim,dropout_rate):
        super(MLP,self).__init__()

        self.num_layers=num_layers
        self.dropout_rate=dropout_rate

        self.lins=nn.ModuleList()
        self.lins.append(nn.Linear(input_dim,hidden_unit))
        for i in range(num_layers-2):
            self.lins.append(nn.Linear(hidden_unit,hidden_unit))
        self.lins.append(nn.Linear(hidden_unit,output_dim))

        self.bns=nn.ModuleList()
        for i in range(num_layers-1):
            self.bns.append(nn.BatchNorm1d(hidden_unit))
        self.bns.append(nn.BatchNorm1d(output_dim))
    
    def forward(self,x):
        for i in range(self.num_layers-1):
            x=self.lins[i](x)
            x=self.bns[i](x)
        x=self.lins[self.num_layers-1](x)
        
        return {'out':F.log_softmax(x, dim=1),'emb':x}
        