import itertools
import os
import os.path as osp
import pickle
import urllib3
from collections import namedtuple
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

# 保存处理好的Cora数据,论文类别及其引用关系
Data = namedtuple('Data', ['x', 'y', 'adjacency', 'train_mask', 'val_mask', 'test_mask'])


class CoraData(object):
    download_url = "https://github.com/kimiyoung/planetoid/raw/master/data"
    filenames = ["ind.cora{}".format(name) for name in ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]

    def __init__(self, data_root="cora", rebuild=False):
        """ 数据下载，处理，加载
        :param data_root: 根目录，原始数据路径：data_root/raw;缓存处理数据：data_root/processed_cora.pkl
        :param rebuild: 是否重建数据集
        """
        self.data_root = data_root
        save_file = osp.join(self.data_root, "processed_cora.pkl")
        if osp.exists(save_file) and not rebuild:
            print("Using Cached file: {}".format((save_file)))
            self._data = pickle.load(open(save_file, "rb"))
        else:
            self.maybe_download()
            self._data = self.process_data()
            with open(save_file, "wb") as f:
                pickle.dump(self.data, f)
            print("Cache file: {}".format((save_file)))

    @property
    def data(self):
        return self._data

    def maybe_download(self):
        save_path = osp.join(self.data_root, "raw")
        for name in self.filenames:
            if not osp.exists(osp.join(save_path, name)):
                self.download_data("{}/{}".format(self.download_url, name), save_path)

    @staticmethod
    def download_data(url, save_path):
        if not osp.exists(save_path):
            os.makedirs(save_path)
        http = urllib3.PoolManager()
        r = http.request('GET', url)
        data = r.data

        filename = osp.basename(url)

        with open(osp.join(save_path, filename), "wb") as f:
            f.write((data.read()))
        return True

    def process_data(self):
        # 处理数据，的到节点特征和标签，邻接矩阵，训练集、验证集以及测试集
        print("Processed data...")
        _, tx, allx, y, ty, ally, graph, test_index = [self.read_data(osp.join(self.data_root, "raw", name)) for
                                                       name in self.filenames]
        train_index = np.arange(y.shape[0])
        val_index = np.arange(y.shape[0], y.shape[0] + 500)
        sorted_test_index = sorted(test_index)

        x = np.concatenate((allx, tx), axis=0)
        y = np.concatenate((ally, ty), axis=0).argmax(axis=1)

        x[test_index] = x[sorted_test_index]
        y[test_index] = y[sorted_test_index]
        num_nodes = x.shape[0]

        train_mask = np.zeros(num_nodes, dtype=np.bool)
        val_mask = np.zeros(num_nodes, dtype=np.bool)
        test_mask = np.zeros(num_nodes, dtype=np.bool)
        train_mask[train_index] = True
        val_mask[train_index] = True
        test_mask[train_index] = True
        adjacency = self.build_adjacency(graph)
        print("Node's feature shape:", x.shape)
        print("Node's label shape:", y.shape)
        print("Adjacency shape:", adjacency.shape)
        print("Number of training nodes: ", train_mask.sum())
        print("Number of validation nodes: ", val_mask.sum())
        print("Number of test nodes: ", test_mask.sum())

        return Data(x=x, y=y, adjacency=adjacency, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    @staticmethod
    def build_adjacency(adj_dict):
        # 根据邻接表创建邻接矩阵
        edge_index = []
        num_nodes = len(adj_dict)
        for src, dst in adj_dict.items():
            edge_index.extend([src, v] for v in dst)
            edge_index.extend([v, src] for v in dst)
        # 存在重复边，删除
        edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
        edge_index = np.asarray(edge_index)
        adjacency = sp.coo_matrix((np.ones(len(edge_index)), (edge_index[:, 0], edge_index[:, 1])),
                                  shape=(num_nodes, num_nodes), dtype="float32")
        return adjacency

    @staticmethod
    def read_data(path):
        """ 读取原始数据 """
        name = osp.basename(path)
        if name == "ind.cora.test.index":
            out = np.genfromtxt(path, dtype="int64")
            return out
        else:
            out = pickle.load(open(path, "rb"), encoding="latin1")
            out = out.toarray() if hasattr(out, "toarray") else out
            return out


class GraphConvolution(nn.Module):
    def __init__(self,input_dim,output_dim,use_bias=True):
        # 图卷积：Lsym*X*W
        super(GraphConvolution,self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.use_bias=use_bias
        self.weight=nn.Parameter(torch.Tensor(input_dim,output_dim))
        if self.use_bias:
            self.bias=nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self,adjacency,input_feature):
        # 使用稀疏乘法
        support=torch.mm(input_feature,self.weight)
        outpot=torch.sparse.mm(adjacency,support)
        if self.use_bias:
            outpot+=self.bias
        return outpot


class GCcnNet(nn.Module):
    def __init__(self,input_dim=1433):
        super(GCcnNet,self).__init__()
        self.gcn1=GraphConvolution(input_dim,16)
        self.gcn2 = GraphConvolution(16, 7)

    def forward(self,adjacency,feature):
        h=F.relu(self.gcn1(adjacency,feature))
        logits=self.gcn2(adjacency,h)
        return logits


def normalization(adjacency):
    # 计算D^-0.5*A*D^-0.5
    adjacency += sp.eye(adjacency.shape[0]) # 自连接
    degree=np.array(adjacency.sum(1))
    d_hat=sp.diags((np.power(degree,-0.5).flatten()))
    return d_hat.dot(adjacency).dot(d_hat).tocoo()


learning_rate =0.1
weight_decay=5e-4
epochs=30
model=GCcnNet()
criterion=nn.CrossEntropyLoss
optimizer=optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
# load data
dataset=CoraData().data
x= dataset.x / dataset.x.sum(1,keepdims=True)
tensor_x=torch.from_numpy(x)
tensor_y=torch.from_numpy(dataset.y)
tensor_train_mask=torch.from_numpy(dataset.train_mask)
tensor_val_mask=torch.from_numpy(dataset.val_mask)
tensor_test_mask=torch.from_numpy(dataset.test_mask)
normalize_adjacency=normalization(dataset.adjacency)

indices=torch.from_numpy(np.asarray([normalize_adjacency.row,normalize_adjacency.col]).astype('int64')).long()
values=torch.from_numpy(normalize_adjacency.data.astypr(np.float32))
tensor_adjacency=torch.sparse.FloatTensor(indices,values,(2708,2708))


def train():
    loss_history=[]
    val_acc_history=[]
    model.train()
    train_y=tensor_y[tensor_train_mask]
    for epoch in range(epochs):
        logits=model(tensor_adjacency,tensor_x)
        train_mask_logits=logits[tensor_train_mask]
        loss = criterion(train_mask_logits,train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc=eval(tensor_train_mask)
        val_acc=eval(tensor_val_mask)

        loss_history.append(loss.item())
        val_acc_history.append(val_acc.item())
        print("Epoch {:03d}: Loss {:.4f},TrainAcc {:.4},ValAcc {:.4f}".format(epoch,loss.item(),train_acc.item(),val_acc.item()))

    return loss_history,val_acc_history


def eval(mask):
    model.eval()
    with torch.no_grad():
        logits=model(tensor_adjacency,tensor_x)
        test_mask_logits=logits[mask]
        predict_y=test_mask_logits.max(1)[1]
        accuracy=torch.eq(predict_y,tensor_y[mask]).float().mean()

    return accuracy


train()



