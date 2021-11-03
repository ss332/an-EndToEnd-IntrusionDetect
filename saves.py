import numpy as np
import pandas as pd
import time
import re
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scapy import all
import os

# -A "2018-03-02 22:11:00"  -B "2018-03-02 23:34:00"
# -A "2018-03-03 02:24:00"  -B "2018-03-03 03:55:00"

# -A "2018-02-21 22:09:00"  -B "2018-02-21 22:43:00"
# -A "2018-02-22 02:05:00"  -B "2018-02-22 03:05:00"
# 将格式字符串转换为时间戳
a_start = "Mar 02 22:11:00 2018"
a_end = "Mar 02 23:34:00 2018"
c_start = "Mar 03 02:24:00 2018"
c_end = "Mar 03 03:55:00 2018"

b_start = "Feb 21 22:08:50 2018"
b_end = "Feb 21 22:43:00 2018"
d_start = "Feb 22 02:05:00 2018"
d_end = "Feb 22 03:05:00 2018"

print(time.mktime(time.strptime(a_start, "%b %d %H:%M:%S %Y")))
print(time.mktime(time.strptime(a_end, "%b %d %H:%M:%S %Y")))
print(time.mktime(time.strptime(c_start, "%b %d %H:%M:%S %Y")))
print(time.mktime(time.strptime(c_end, "%b %d %H:%M:%S %Y")))

print(time.mktime(time.strptime(b_start, "%b %d %H:%M:%S %Y")))
print(time.mktime(time.strptime(b_end, "%b %d %H:%M:%S %Y")))
print(time.mktime(time.strptime(d_start, "%b %d %H:%M:%S %Y")))
print(time.mktime(time.strptime(d_end, "%b %d %H:%M:%S %Y")))

time1 = [1519999860.0, 1520004840.0]
time2 = [1520015040.0, 1520020500.0]

time3 = [1519222130.0, 1519224180.0]
time4 = [1519236300.0, 1519239900.0]
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(1519222140.0)))

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

# for i in range(100001):
#     path = r'E:\space\space2{}.pt'.format(i)
#     if os.path.exists(path):  # 如果文件存在
#         # 删除文件，可使用以下两种方法。
#         os.remove(path)
#         # os.unlink(path)
#     else:
#         print('no such file:%s' % path)  # 则返回文件不存在

record_name = r'E:\ddos\ddos{}.pt'.format(26)
# 归一化
record_flow = torch.load(record_name) / 255.0
space_tensor = torch.zeros(784)
size = 0
for k in range(record_flow.size(0)):
    x = torch.flatten(record_flow[k])
    x1 = torch.flatten(record_flow[k], 0)
    print(x == x1)