import numpy as np
import time
import re
# packets = all.rdpcap(r'E:\Amazon\DDOS-LOIC-UDP.pcap')
# print(packets[131].time)
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(packets[131].time)))
# print(packets[131].show())
i=7
text_name = '../bot/array{}.txt'.format(i)
print(np.loadtxt(text_name))