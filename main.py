import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
from scapy import all
import decimal
import zlib

if __name__ == "__main__":

    # ids = [6686, 52661, 84092, 16, 17, 18, 33798, 42837, 36156]
    #
    # label = ['bot', 'loic-udp', 'hoic']
    #
    # figure = plt.figure(figsize=(8, 8))
    # a = 0
    # for i in range(9):
    #     figure.add_subplot(5, 3, i + 1)
    #     a = i // 3
    #     if a == 0:
    #         space_img = r'E:\space\space{}.pt'.format(ids[i])
    #     else:
    #         space_img = r'E:\space2\space{}.pt'.format(ids[i])
    #     s_img = torch.load(space_img).view(28, -1)
    #     plt.title(label[a])
    #     plt.axis("off")
    #     plt.imshow(s_img.squeeze(), cmap="gray")
    # plt.savefig("visual.png")
    # plt.show()

    flows = pd.read_csv('flows2.csv')
    print(flows[flows['label'] == 0].shape)
    print(flows[flows['label'] == 1].shape)
    print(flows[flows['label'] == 2].shape)


