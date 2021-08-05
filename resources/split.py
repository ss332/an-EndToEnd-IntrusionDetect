import numpy as np
import matplotlib.pyplot as plt
from scapy import all
import time
import operator
import re
import pandas as pd
import math


n_iters = 100000
print_every = 5000

# 切割tcp流：五元组(SourceIP, DestinationIP, SourcePort, DestinationPort, Protocol)
# "2018-03-02 22:11:00"  -B "2018-03-03 08:34:00"
# attack = 18.219.211.138
# --------------------------------------------------------
# 目的mac地址6|源mac地址6|类型2|数据         46-1500|FCS4    |
# --------------------------------------------------------
# 版本|首部长度|    区分服务1     |总长度2                    |
#        标识 2                |标志3bit|片偏移             |
# 生存时间2      |协议2          |首部校验和2                 |
#                          源地址4                         |
#                          目的地址4                       |
#                     可选字段(长度可变) 填充                |
# --------------------------------------------------------
# 以太网数据帧格式：目的mac地址6，源mac地址6，协议类型2，数据（14字节）
# ip数据格式（20），在源IP地址4，目的IP地址4，之前有12字节，因此混淆这两个数据需要（在[27,34])之间填0
# 无论tcp（20）数据包还是udp（8）数据包2，源端口2，目的端口都紧随其后，因此最后是在[27,38]之间填0

packets = all.PcapReader(r'E:\Amazon\DDOS-LOIC-UDP.pcap')


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


flows = []
flows_time = []
flowId = 0

start = time.time()
print('读取开始:')
for i in range(n_iters):
    try:
        packet = packets.read_packet()
    except:
        break
    isSyn = 0
    isFin = 0
    isTimeOut = 0
    if packet.haslayer('IP'):
        if packet.haslayer('TCP'):
            flags = str(packet['TCP'].flags)
            if re.search('S', flags) is not None:
                isSyn = 1
            if re.search('F', flags) is not None:
                isFin = 1
            flow = [packet['IP'].src, packet['IP'].dst, packet['TCP'].sport, packet['TCP'].dport, packet['IP'].proto,
                    isFin]
        elif packet.haslayer('UDP'):
            flow = [packet['IP'].src, packet['IP'].dst, packet['UDP'].sport, packet['UDP'].dport, packet['IP'].proto, 0]
    else:
        continue

    tmp_a = []
    size = 0
    for byte in bytes(packet):
        tmp_a.append(byte)
        size = size + 1
        if size == 784:
            break
    if size <= 784:
        while size < 784:
            tmp_a.append(0)
            size = size + 1

    tmp_a = np.array(tmp_a, dtype=np.int64)
    arrays = tmp_a.reshape((1,28,28), order='C')
    file_name = '../bot/FlowId{}.npy'.format(flowId)
    if i == 0:
        flows.append(flow)
        flows_time.append(packet.time)
        np.save(file_name, arrays)
        print('完成flow', flowId)
        flowId = flowId + 1
    else:
        for j in range(len(flows) - 1, -1, -1):
            if (packet.time - flows_time[j]) >= 600:
                isTimeOut = 1
            if isSyn == 1:
                flows.append(flow)
                flows_time.append(packet.time)
                np.save(file_name, arrays)
                flowId = flowId + 1
                break
            elif operator.eq(flows[j], flow) and isTimeOut != 1 and flows[j][5] != 1:
                flows[j][5] = isFin
                record_name = '../bot/FlowId{}.npy'.format(j)
                record_flow = np.load(record_name)
                merge_flow = np.concatenate((record_flow, arrays), axis=0)
                np.save(record_name, merge_flow)
                # print('flow %d 合并 flow %d' % (i, j))
                break
            elif j == 0 and not operator.eq(flows[j], flow):
                flows.append(flow)
                flows_time.append(packet.time)
                np.save(file_name, arrays)
                flowId = flowId + 1

    if (i+1) % print_every == 0:
        print('%d %d%% (%s)  %s' % (i+1, (i+1) / n_iters * 100, timeSince(start), flow))


df = pd.DataFrame(flows, columns=['SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort', 'Protocol', 'isFin'])
df.to_csv('flows.csv')
packets.close()





