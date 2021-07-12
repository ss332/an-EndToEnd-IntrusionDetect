import numpy as np
import matplotlib.pyplot as plt
from scapy import all
import time
import operator
import re

# 切割tcp流：六元组（FlowId,SourceIP, DestinationIP, SourcePort, DestinationPort, Protocol)
# E:\Amazon\origin\pcap\UCAP172.31.69.25 print (time.mktime(time.strptime(a,"%Y-%m-%d %H:%M:%S"))) editcap -A
# "2018-03-02 22:11:00"  -B "2018-03-03 08:34:00"  E:\Amazon\origin\pcap\capEC2AMAZ-O4EL3NG-172.31.69.23
# E:\Amazon\DDOS-LOIC-UDP.pcap
# attack = 18.219.211.138

packets = all.rdpcap(r'E:\Amazon\DDOS-LOIC-UDP.pcap')
print(packets[51618].time)
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(packets[51618].time)))
print(packets[51618].show())

flows = []
flows_time = []
packets = all.rdpcap(r'E:\Amazon\DDOS-LOIC-UDP.pcap')
i = 0
print(packets[0])

for packet in packets:
    isSyn = 0
    isFin = 0
    isTimeOut = 0

    if packet.haslayer('TCP'):
        flow = [i, packet['IP'].src, packet['IP'].dst, packet['TCP'].sport, packet['TCP'].sport, packet['IP'].proto]
        if re.search('S', packet['TCP'].flags) is not None:
            isSyn = 1
        if re.search('F', packet['TCP'].flags) is not None:
            isFin = 1

    elif packet.haslayer('UDP'):
        flow = [i, packet['IP'].src, packet['IP'].dst, packet['UDP'].sport, packet['UDP'].sport, packet['IP'].proto]
    else:
        flow = []

    tmp_a = []
    size = 0
    for byte in packets:
        tmp_a.append(byte)
        if size <= 256:
            size = size + 1
        else:
            break
    if size <= 256:
        while size <= 256:
            tmp_a.append(0)

    tmp_a = np.array(tmp_a, dtype=np.uint8)
    arrays = tmp_a.reshape((18, 18), order='C')
    text_name = '../bot/FlowId{}.txt'.format(i)
    if i == 0:
        flows.append(flow)
        flows_time.append(packet.time)
        np.savetxt(text_name, arrays)
    else:
        if (packet.time - flows_time[i]) >= 600:
            isTimeOut = 1
        if operator.eq(flows[i - 1], flow):
            try:
                record_flow = np.loadtxt(text_name)
                np.concatenate((record_flow, arrays), axis=0)
                np.savetxt(text_name, arrays)
            except OSError:
                print('新的流，正在创建：')
            else:
                np.savetxt(text_name, arrays)

    if i % 1000 == 0:
        print("已完成：i")
