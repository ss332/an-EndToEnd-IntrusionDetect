import numpy as np
import pandas as pd
import time
import re
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scapy import all
import os

dates = ["2018-02-14 22:32:00", "2018-02-15 00:09:00", "2018-02-15 02:01:00", "2018-02-15 03:31:00",
         "2018-02-15 21:26:00", "2018-02-15 22:09:00", "2018-02-15 22:59:00", "2018-02-15 23:40:00",
         "2018-02-16 22:12:00", "2018-02-16 23:08:00", "2018-02-17 01:45:00", "2018-02-17 02:19:00",
         "2018-02-20 22:12:00", "2018-02-20 23:17:00", "2018-02-21 01:32:00",
         "2018-02-21 22:09:00", "2018-02-21 22:43:00", "2018-02-22 02:05:00", "2018-02-22 03:05:00",
         "2018-02-22 22:17:00", "2018-02-22 23:24:00", "2018-02-23 01:50:00", "2018-02-23 02:29:00",
         "2018-02-23 04:15:00", "2018-02-23 04:29:00",  #6
         "2018-02-28 22:50:00", "2018-03-01 00:05:00", "2018-03-01 01:42:00", "2018-03-01 02:40:00",
         "2018-03-02 22:11:00", "2018-03-02 23:34:00", "2018-03-03 02:24:00", "2018-03-03 03:55:00"

         ]


def get_second(dates):
    i = 1
    for shi in dates:
        print(i,shi,':',time.mktime(time.strptime(shi, "%Y-%m-%d %H:%M:%S")))
        i=i+1


get_second(dates)

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(1519060380.14541)))

# flows = pd.read_csv('flows_ddos.csv')
# flows = flows.iloc[0:106691,:]
#
# flows.to_csv('flows_ddos.csv')

# flows1 = pd.read_csv('./resources/flows_benign.csv')
# for i in range(flows1.shape[0]):
#     flows1['id'][i]+=50000
# flows2 = pd.read_csv('./resources/flows_ddos2.csv')
# bot_ddos = pd.concat([flows2, flows1], axis=0)

# bot_ddos=pd.read_csv('resources/flows_ddos.csv')
# bot_ddos.drop('se', axis=1, inplace=True)
# bot_ddos.to_csv('flows_ddos.csv')
# editcap -A "2018-02-17 01:45:00"  -B "2018-02-17 02:19:00" H:\ids2018\fri-16-02\UCAP172.31.69.25-part1.pcap
for i in range(100002):
    path = r'D:\sessions\files3\session{}.pt'.format(i)
    if os.path.exists(path):  # 如果文件存在
        # 删除文件，可使用以下两种方法。


        os.remove(path)

        # os.unlink(path)
    else:
        print('no such file:%s' % path)  # 则返回文件不存在
