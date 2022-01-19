import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence


class ArcNet(nn.Module):
    # 整体网络

    def __init__(self):
        super(ArcNet, self).__init__()
        packet_features = 512  # q
        session_features = 512  # v
        hidden_size = 512

        glimpses = 2
        self.image = SessionProcessor(drop=0.5)
        self.text = PacketProcessor(drop=0.5)

        self.attention = Attention(v_features=512, q_features=512, mid_features=256, glimpses=2, drop=0.5, )

        self.classifier = Classifier(in_features=glimpses * 512 + 512, mid_features=512, categorys=15, drop=0.5, )

        # 权重0初始化
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, q, q_len):

        v = self.image(v)
        # q=[n,hiddensize]
        q = self.text(q, list(q_len.data))
        # 归一化 v=[n,c,w,h]
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
        a = self.attention(v, q)  # [n,g,w,h]
        v = apply_attention(v, a)  # [n,g*v]

        combined = torch.cat([v, q], dim=1)  # [n,g*v+q]
        detect = self.classifier(combined)
        return detect


class SessionProcessor(torch.nn.Module):
    def __init__(self, drop=0.0):
        super(SessionProcessor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, (3, 3))
        self.conv3 = nn.Conv2d(64, 128, (3, 3))
        self.conv4 = nn.Conv2d(128, 512, (1, 1))
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # [1,32,32]-> [32,30,30]->[32,15,15]
        x = self.pool(self.relu(self.conv1(x)))
        # [1,15,15]-> [64,13,13]->[64,6,6]
        x = self.pool(self.relu(self.conv2(x)))
        # [64,6,6]->[128,4,4]
        x = self.relu(self.conv3(x))
        # [512,4,4]->[512,4,4]
        x = self.relu(self.conv4(x))
        x = self.drop(x)
        # 512*4*4=8192

        return x


class PacketProcessor(nn.Module):
    def __init__(self, embedding_features=320, hidden_size=512, drop=0.0):
        super(PacketProcessor, self).__init__()

        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()

        self.gru = nn.GRU(input_size=embedding_features,
                          hidden_size=hidden_size,
                          num_layers=1)
        self.features = hidden_size
        # 初始化参数
        self._init_gru(self.gru.weight_hh_l0)
        self.gru.bias_hh_l0.data.zero_()

        # init.xavier_uniform(self.embedding.weight)

    def _init_gru(self, weight):
        for w in weight.chunk(4, 0):
            init.xavier_uniform_(w)

    def forward(self, q, q_len):
        embedded = q
        tanhed = self.tanh(self.drop(embedded))
        packed = pack_padded_sequence(tanhed, q_len, batch_first=True,enforce_sorted=False)
        output, hidden = self.gru(packed)  # hidden=[1,n,512]
        return hidden.squeeze(0)  # [N,512]


# 两层全连接,映射到输出空间维度
class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, categorys, drop=0.0):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_features, mid_features)
        self.fc2 = nn.Linear(mid_features, categorys)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()

    def forward(self, combine):
        # 26*26 13
        c = self.drop(combine)
        c = self.relu(self.fc1(c))
        c = self.drop(c)
        c = self.fc2(c)

        return c


# v_features=512, q_features=512, mid_features=256
class Attention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention, self).__init__()
        self.v_conv = nn.Conv2d(v_features, mid_features, (1, 1), bias=False)  # let self.lin take care of bias
        self.q_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, glimpses, (1, 1))

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, q):
        # v=[n,256,14,14]
        v = self.v_conv(self.drop(v))
        # q=[n,256]
        q = self.q_lin(self.drop(q))
        # q=
        q = tile_2d_over_nd(q, v)
        x = self.relu(v + q)
        x = self.x_conv(self.drop(x))  # [n,2,14,14]

        return x


def tile_2d_over_nd(feature_vector, feature_map):
    # 在特征图上加上所有数据包特征向量，特征向量需要与特征图的批次与通道数一样
    n, c = feature_vector.size()
    spatial_size = feature_map.dim() - 2  # 2
    # tile=[n,c,w,h]
    tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)
    return tiled


def apply_attention(input, attention):
    # 在输入特征图上应用注意力权重
    n, c = input.size()[:2]
    # glimpses=2
    glimpses = attention.size(1)

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input = input.view(n, 1, c, -1)  # [n, 1, c, s]
    attention = attention.view(n, glimpses, -1)  # [n,g,s]
    attention = F.softmax(attention, dim=-1).unsqueeze(2)  # [n, g, 1, s]
    weighted = attention * input  # [n, g, v, s]
    weighted_mean = weighted.sum(dim=-1)  # [n, g, v]
    return weighted_mean.view(n, -1)
