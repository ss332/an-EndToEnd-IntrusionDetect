import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
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

# feature_vector=torch.tensor([0,1,2,3,4,5,6,7,8,9]).view(1,-1)
# input=torch.randn((1,10,4,4))
# # tile=tile_2d_over_nd(feature_vector,feature_map)
# #
# # print(tile.shape)
# # print(tile)
# # print(feature_map+tile)
# n, c = input.size()[:2]
# input = input.view(n, 1, c, -1)  # [n, 1, c, s]=
# attention=torch.randn((1,2,16))
# attention = F.softmax(attention, dim=-1).unsqueeze(2)  # [n, g, 1, s]=[n,2,1,16]
# weighted = attention * input  # [n, g, v, s] [n,g,1,s]*[n,1,c,s]=[n,g,,16]
# weighted_mean = weighted.sum(dim=-1)  # [n, g, v]
# print(input.shape)
# print(weighted.shape)
# print(weighted_mean.shape)
# print(weighted_mean)

# a=torch.zeros((5,320))
# c=torch.zeros((32,320))
# b=a.expand_as(c)
# print(b.shape)
# 在输入特征图上应用注意力权重


# attention = np.random.randint(0,14,(15,15))
# sentence=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(attention, cmap='bone')
# fig.colorbar(cax)
# fontdict = {'fontsize': 12}
# ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
# ax.set_yticklabels([''] + sentence, fontdict=fontdict)
# ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
# ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
# plt.ylabel('Actual label')
# plt.xlabel('Predict label')
# plt.suptitle('Attention weights')
# plt.show()

from matplotlib.colors import LinearSegmentedColormap
from captum.attr import (
    IntegratedGradients,
    LayerIntegratedGradients,
    TokenReferenceBase,
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
    visualization
)


def encode_sessions(session):
    session_tensor = torch.zeros(1024)
    size = 0
    for k in range(session.size(0)):
        x = session[k]
        for j in range(x.size(0)):
            if x[j] != 0 and size < 1024:
                session_tensor[size] = x[j]
                size = size + 1

        # (n,c,w,h)
    session_tensor = session_tensor.view(1, 1, 32, 32)
    return session_tensor
default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                 [(0, '#ffffff'),
                                                  (0.25, '#252b36'),
                                                  (1, '#000000')], N=256)

file = r'C:\sessions\files{}\session{}.pt'.format(7, 10516)
orin=torch.load(file)
orin = encode_sessions(orin)
text=torch.load('./s13.pt')
print()
original_im_mat = np.transpose(orin.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
attributions_img = np.transpose(text.unsqueeze(0).cpu().detach().numpy(), (1, 2, 0))

visualization.visualize_image_attr_multiple(attributions_img, original_im_mat,
                                            ["original_image", "heat_map"], ["all", "absolute_value"],
                                            titles=["Session", 'attribution-Infiltration'],
                                            cmap=default_cmap,
                                            show_colorbar=True)