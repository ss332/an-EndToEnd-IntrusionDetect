import numpy as np
import matplotlib.pyplot as plt
from scapy import all
import time
import operator
import re
import pandas as pd
import math

n_iters = 100000
print_every = 1000

# 切割tcp流：六元组（FlowId,SourceIP, DestinationIP, SourcePort, DestinationPort, Protocol)
# "2018-03-02 22:11:00"  -B "2018-03-03 08:34:00"  E:\Amazon\origin\pcap\capEC2AMAZ-O4EL3NG-172.31.69.23
# attack = 18.219.211.138

packets = all.PcapReader(r'E:\Amazon\DDOS-LOIC-UDP.pcap')



def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


flows = []
flows_time = []
i = 0

start = time.time()
print('读取开始:')
for i in range(1):
    packet = packets.read_packet()
    isSyn = 0
    isFin = 0
    isTimeOut = 0

    if packet.haslayer('TCP'):
        flags = str(packet['TCP'].flags)
        print(flags)
        if re.search('S', flags) is not None:
            isSyn = 1
        if re.search('F', flags) is not None:
            isFin = 1
        flow = [i, packet['IP'].src, packet['IP'].dst, packet['TCP'].sport, packet['TCP'].dport, packet['IP'].proto,
                isFin]
    elif packet.haslayer('UDP'):
        flow = [i, packet['IP'].src, packet['IP'].dst, packet['UDP'].sport, packet['UDP'].dport, packet['IP'].proto, 0]
    else:
        continue

    tmp_a = []
    size = 0
    for byte in bytes(packet):
        tmp_a.append(byte)
        size=size+1
        if size == 256:
            break
    if size <= 256:
        while size <256:
            tmp_a.append(0)
            size=size+1

    tmp_a = np.array(tmp_a, dtype=np.uint8)
    arrays = tmp_a.reshape((16, 16), order='C')
    print(arrays)
    text_name = '../bot/FlowId{}.txt'.format(i)
    if i == 0:
        flows.append(flow)
        flows_time.append(packet.time)
        np.savetxt(text_name, arrays)
        print('完成flow', i)
        i = i + 1
    else:
        for j in range(len(flows) - 1, -1, -1):
            if (packet.time - flows_time[j]) >= 600:
                isTimeOut = 1
            if isSyn == 1:
                flows.append(flow)
                flows_time.append(packet.time)
                np.savetxt(text_name, arrays)
                i = i + 1
                break
            elif operator.eq(flows[j], flow) and isTimeOut != 1 and flows[j][6] != 1:
                try:
                    record_flow = np.loadtxt(text_name)
                    np.concatenate((record_flow, arrays), axis=0)
                    np.savetxt(text_name, arrays)
                except OSError:
                    print('新的流，正在创建：')
                else:
                    flows.append(flow)
                    flows_time.append(packet.time)
                    np.savetxt(text_name, arrays)
                    print('完成flow', i)
                    i = i + 1
                break
            elif j == 0:
                flows.append(flow)
                flows_time.append(packet.time)
                np.savetxt(text_name, arrays)
                print('完成flow', i)
                i = i + 1

    if i-1 == 1:
        print(
            '%d %d%% (%s)  %s' % (i-1, i-1 / n_iters * 100, timeSince(start), flow))
    if i == 102:
        break

df = pd.DataFrame(flows,
                  columns={'FlowId', 'SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort', 'Protocol', 'isFin'})
df.to_csv('flows.csv')
packets.close()
