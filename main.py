
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
    # flows = pd.read_csv('flows_ddos.csv')

    # ids=[66863,66865,66867,9011,43841,106687,15, 3, 7,23142,51806,104676]
    #
    # label=['benign','bot','loic-udp','hoic','benign']
    #
    # figure = plt.figure(figsize=(8, 8))
    # a=0
    # for i in range(12):
    #     figure.add_subplot(5, 3, i+1)
    #     a=i//3
    #     if a<2:
    #         space_img = r'E:\space\spaceId{}.pt'.format(ids[i])
    #     else:
    #         space_img = r'E:\space2\spaceId{}.pt'.format(ids[i])
    #     s_img = torch.load(space_img).view(28, -1)
    #     plt.title(label[a])
    #     plt.axis("off")
    #     plt.imshow(s_img.squeeze(), cmap="gray")
    # plt.savefig("visual.png")
    # plt.show()
    flows = pd.read_csv('resources/flows_ddos.csv')
    print(flows[flows['label'] == 1].shape)
    print(flows[flows['label'] == 2].shape)

    df2 = pd.read_csv(r"E:\data\CICIDS2018\Wednesday-21-02-2018_TrafficForML_CICFlowMeter.txt")
    benign_df2= df2[df2["Label"] == 'DDOS attack-HOIC']
    print(benign_df2.shape)









