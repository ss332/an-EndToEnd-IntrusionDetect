import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
from scapy import all
import decimal
import zlib
import seaborn as sns

if __name__ == "__main__":

    ids = [11217,22432,29953,43350,67527,50970]
    ids2 = [2430, 4218, 24732, 28923, 29890, 31629,27442,30203,32629]
    ids3=[281,8423,13386,15187,17086,27665]

    ids45= [21928, 35504, 48418,77, 90, 154, 2300, 53075, 91768]
    ids6 = [362, 372, 383, 991, 1043, 1007,1391,1402,1415]

    ids78 = [15424, 38895, 88287, 17968, 47267, 92242]

    label = ['FTP-BruteForce', 'SSH-BruteForce']
    label2 = ['DoS-GoldenEye', 'DoS-Slowloris','DoS-Slowloris']
    label3 = ['DoS-SlowHTTPTest', 'DoS-Hulk']

    label5 = ['DDoS LOIC-HTTP','DDoS-LOIC-UDP', 'DDoS-HOIC']
    label6=['Brute Force-Web','Brute Force-XSS','SQL Injection']

    label78 = ['Infiltration','Bot']

    figure = plt.figure(figsize=(6, 4))
    a = 0
    for i in range(6):
        # if i<3:
        #     flow_name = r'D:\sessions\files7\session{}.pt'.format(ids78[i])
        # else:
        #     flow_name = r'D:\sessions\files8\session{}.pt'.format(ids78[i])
        flow_name = r'D:\sessions\files3\session{}.pt'.format(ids3[i])
        input_tensor = torch.load(flow_name)
        space_tensor = torch.zeros(1024)

        size = 0
        for k in range(input_tensor.size(0)):
            x = input_tensor[k]
            for j in range(x.size(0)):
                if x[j] != 0 and size < 1024:
                    space_tensor[size] = x[j]
                    size = size + 1

        figure.add_subplot(2, 3, i + 1)

        a=i//3
        s_img = space_tensor.view(32, -1)
        plt.title(label3[a])
        plt.axis("off")
        plt.imshow(s_img.squeeze(), cmap="gray")
    plt.savefig("visual3.png")
    plt.show()




