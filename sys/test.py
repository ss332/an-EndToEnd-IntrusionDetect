import torch
import torch.nn as nn
import torch.nn.functional as F
import os


    # rnn = nn.GRU(10, 20, 1)
    # input = torch.randn(5, 3, 10)
    # h0 = torch.randn(1, 3, 20)
    # output, hn = rnn(input, h0)
    # print(output.shape, hn.shape)

def tile_2d_over_nd(feature_vector, feature_map):
    # 在特征图上加上所有数据包特征向量，特征向量需要与特征图的批次与特征维数一样
    n, c = feature_vector.size()
    spatial_size = feature_map.dim() - 2 # 2
    tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)
    return tiled

feature_vector=torch.tensor([0,1,2,3,4,5,6,7,8,9]).view(1,-1)
input=torch.randn((1,10,4,4))
# tile=tile_2d_over_nd(feature_vector,feature_map)
#
# print(tile.shape)
# print(tile)
# print(feature_map+tile)
n, c = input.size()[:2]
input = input.view(n, 1, c, -1)  # [n, 1, c, s]=
attention=torch.randn((1,2,16))
attention = F.softmax(attention, dim=-1).unsqueeze(2)  # [n, g, 1, s]=[n,2,1,16]
weighted = attention * input  # [n, g, v, s] [n,g,1,s]*[n,1,c,s]=[n,g,,16]
weighted_mean = weighted.sum(dim=-1)  # [n, g, v]
print(input.shape)
print(weighted.shape)
print(weighted_mean.shape)
print(weighted_mean)