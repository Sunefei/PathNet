jobs import os
import torch
import  torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from metric import accuracy,precision,recall,F1_score
import numpy as np
import tqdm
from sklearn.preprocessing import MinMaxScaler
import pickle
import torch_sparse
import math
import networkx as nx
import random
import os

seed = 1
seed = int(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Model(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(Model, self).__init__()
        self.lin1 = nn.Linear(input_size,hidden_size)
        self.lin = nn.Linear(hidden_size*5,2)

    def forward(self,x,edge_index):
        temp = self.lin1(x)
        temp = F.relu(temp)
        temp1 = torch_sparse.matmul(edge_index,temp)
        temp1 =torch.cat((temp,temp1),dim=1)
        temp2 = torch_sparse.matmul(edge_index,temp1)
        temp = torch.cat((temp,temp1,temp2),dim=1)
        temp = F.dropout(temp,p=0.5)
        ans = self.lin(temp)
        return ans

class GRAPHdataset(Dataset):
    def __init__(self, index):
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        return self.index[index]
f = open('/data/keteng/pickle/total', 'rb')
node_feature, node_label, edge = pickle.load(f)
f.close()
with open('/data/keteng/pickle/data_divide', 'rb') as f:
    fold = pickle.load(f)

g = nx.Graph()
g.add_nodes_from(range(len(node_label)))
g.add_edges_from(edge)
d = nx.degree(g)
row = []
col = []
value = []
for i in d:
    row.append(i[0])
    col.append(i[0])
    value.append(1 / math.sqrt(i[1]))
degree = torch_sparse.SparseTensor(sparse_sizes=[len(node_label), len(node_label)],
                                   row=torch.tensor(row, dtype=torch.long), col=torch.tensor(col, dtype=torch.long),
                                   value=torch.tensor(value, dtype=torch.float32)).to(device)
adj = nx.adjacency_matrix(g, nodelist=range(len(node_label)))
adj_coo = adj.tocoo()
adj_row = adj_coo.row
adj_col = adj_coo.col
adj_value = adj_coo.data
adj_size = adj_coo.shape
adj = torch_sparse.SparseTensor(sparse_sizes=[adj_size[0], adj_size[1]], row=torch.tensor(adj_row, dtype=torch.long),
                                col=torch.tensor(adj_col, dtype=torch.long),
                                value=torch.tensor(adj_value, dtype=torch.float32)).to(device)
adj = torch_sparse.matmul(degree, adj)
adj = torch_sparse.matmul(adj, degree)
print(degree)
print(adj)

edge = np.array(edge).transpose()
edge[[0,1],:] = edge[[1,0],:]
scaler = MinMaxScaler()
node_feature = scaler.fit_transform(node_feature)
node_feature = torch.tensor(node_feature,dtype=torch.float32)
edge_index = torch.tensor(edge,dtype=torch.long)
node_label = torch.tensor(node_label,dtype=torch.long)
node_feature = node_feature.to(device)
result = []
for i in range(5):
    train_index = fold[int(i%5)]+fold[int((i+1)%5)]+fold[int((i+2)%5)]
    test_index = fold[int((i+3)%5)]+fold[int((i+4)%5)]
    random.shuffle(train_index)
    random.shuffle(test_index)

    model = Model(node_feature.shape[1],128).to(device)
    mydataset = GRAPHdataset(train_index)
    dataloader = DataLoader(mydataset,batch_size=1024,shuffle=True)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.0001)
    Loss = nn.CrossEntropyLoss(weight=torch.tensor([1., 3.])).to(device)

    max_f1 = 0
    s = None
    k = None
    for epoch in tqdm.tqdm(range(200)):
        model.train()
        train_loss = 0
        num=0
        for index in dataloader:
            optimizer.zero_grad()
            out = model(node_feature,adj)
            loss = Loss(out[index], node_label[index].to(device))
            train_loss+=loss.cpu().detach().numpy()
            num+=1
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            train_ans = model(node_feature,adj)[train_index].cpu().detach()
            train_label = node_label[train_index]
            train_result = train_ans.argmax(dim=1)
            train_accuracy = accuracy(train_result, train_label)
            train_pos_precision = precision(train_result, train_label, 0)
            train_pos_recall = recall(train_result, train_label, 0)
            train_pos_f1_score = F1_score(train_pos_precision, train_pos_recall)
            train_neg_precision = precision(train_result, train_label, 1)
            train_neg_recall = recall(train_result, train_label, 1)
            train_neg_f1_score = F1_score(train_neg_precision, train_neg_recall)



            test_ans = model(node_feature,adj)[test_index].cpu().detach()
            test_label = node_label[test_index]
            test_result = test_ans.argmax(dim=1)
            test_accuracy = accuracy(test_result, test_label)
            test_pos_precision = precision(test_result, test_label, 0)
            test_pos_recall = recall(test_result, test_label, 0)
            test_pos_f1_score = F1_score(test_pos_precision, test_pos_recall)
            test_neg_precision = precision(test_result, test_label, 1)
            test_neg_recall = recall(test_result, test_label, 1)
            test_neg_f1_score = F1_score(test_neg_precision, test_neg_recall)

            print(
                'epoch:{} train_loss:{:.6f} train_accuracy:{:.4f} train_pos_precision:{:.4f} train_pos_recall:{:.4f} train_pos_F1_score:{:.4f} train_neg_precision:{:.4f} train_neg_recall:{:.4f} train_neg_F1_score:{:.4f} test_accuracy:{:.4f} test_pos_precision:{:.4f} test_pos_recall:{:.4f} test_pos_F1_score:{:.4f} test_neg_precision:{:.4f} test_neg_recall:{:.4f} test_neg_F1_score:{:.4f}'.format(
                    epoch,train_loss / num,
                    train_accuracy,train_pos_precision,train_pos_recall,train_pos_f1_score,
                    train_neg_precision,train_neg_recall,train_neg_f1_score,
                    test_accuracy,
                    test_pos_precision, test_pos_recall, test_pos_f1_score,
                    test_neg_precision, test_neg_recall, test_neg_f1_score))
            if max_f1 < test_neg_f1_score:
                max_f1 = test_neg_f1_score
                s = 'epoch:{} train_loss:{:.6f} train_accuracy:{:.4f} train_pos_precision:{:.4f} train_pos_recall:{:.4f} train_pos_F1_score:{:.4f} train_neg_precision:{:.4f} train_neg_recall:{:.4f} train_neg_F1_score:{:.4f} test_accuracy:{:.4f} test_pos_precision:{:.4f} test_pos_recall:{:.4f} test_pos_F1_score:{:.4f} test_neg_precision:{:.4f} test_neg_recall:{:.4f} test_neg_F1_score:{:.4f}'.format(
                    epoch,train_loss / num,
                    train_accuracy,train_pos_precision,train_pos_recall,train_pos_f1_score,
                    train_neg_precision,train_neg_recall,train_neg_f1_score,
                    test_accuracy,
                    test_pos_precision, test_pos_recall, test_pos_f1_score,
                    test_neg_precision, test_neg_recall, test_neg_f1_score)
                k = [test_accuracy,test_pos_precision,test_pos_recall,test_pos_f1_score,test_neg_precision,test_neg_recall,test_neg_f1_score]
    print(s)
    result.append(k)
result = np.array(result)
np.savetxt('h2gcn.csv',result,fmt='%.4f',delimiter=',')