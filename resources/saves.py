import numpy as np
import pandas as pd
import time
import re
# packets = all.rdpcap(r'E:\Amazon\DDOS-LOIC-UDP.pcap')
# print(packets[131].time)
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(packets[131].time)))
# print(packets[131].show())

for j in range(10,-1,-1):
    print(j)
a=[]
b=[1,2,3,4,5,6]
c=['a',3,'r',2,4,'d']
a.append(b)
a.append(c)
print(a[1][3])

df=pd.read_csv('flows.csv')
print(df)