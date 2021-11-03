from scapy import all
import time
import operator
import re
import pandas as pd
import math
import torch



n_iters = 246000
print_every = 5000

files = [r'E:\Amazon\capBot-23.pcap',
         r'E:\Amazon\capBot-17.pcap',
         r'E:\Amazon\capBot-8.pcap',
         r'E:\Amazon\capBot-12.pcap',
         r'E:\Amazon\capBot-29.pcap',
         r'E:\Amazon\capBot-30.pcap']

files2 = [r'E:\Amazon\DDOS-LOIC-UDP.pcap',
          r'E:\Amazon\DDOS-HOIC.pcap']

files3=[r'E:\Amazon\wednes-21-02\pcap\capEC2AMAZ-O4EL3NG-172.31.69.23',
        r'E:\Amazon\wednes-21-02\pcap\capEC2AMAZ-O4EL3NG-172.31.69.26',
        r'E:\Amazon\wednes-21-02\pcap\capEC2AMAZ-O4EL3NG-172.31.69.29',
        r'E:\Amazon\wednes-21-02\pcap\capEC2AMAZ-O4EL3NG-172.31.69.30']


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


flows = []
flows_time = []
flowId = 0
sequence = 0

start = time.time()
print('读取开始......')
for file in files3:
    sequence += 1
    packets = all.PcapReader(file)
    print('file', sequence)
    hasnext = True
    i = 0
    while (hasnext):
        try:
            packet = packets.read_packet()
        except:
            packets.close()
            hasnext = False
            continue
        i = i + 1
        if i >= 1000000:
            i = 2
        isSyn = 0
        isFin = 0
        isTimeOut = 0

        if packet.haslayer('IP'):
            if packet.haslayer('TCP'):
                flags = str(packet['TCP'].flags)
                if re.search('SEC', flags) is not None:
                    isSn = 1
                if re.search('F', flags) is not None:
                    isFin = 1

                flow = [packet['IP'].proto, packet['TCP'].sport, packet['TCP'].dport, packet['IP'].src,
                        packet['IP'].dst, 0]
                re_flow = [packet['IP'].proto, packet['TCP'].dport, packet['TCP'].sport, packet['IP'].dst,
                           packet['IP'].src, 0]

            elif packet.haslayer('UDP'):

                flow = [packet['IP'].proto, packet['UDP'].sport, packet['UDP'].dport, packet['IP'].src,
                        packet['IP'].dst, 0]
                re_flow = [packet['IP'].proto, packet['UDP'].dport, packet['UDP'].sport, packet['IP'].dst,
                           packet['IP'].src, 0]
        else:
            continue

        tmp_a = []
        size = 0

        for byte in bytes(packet):
            # if 26 < size < 39:
            #     tmp_a.append(0)
            # else:
            tmp_a.append(byte)
            size = size + 1
            if size == 256:
                break
        if size <= 256:
            while size < 256:
                tmp_a.append(0)
                size = size + 1

        tmp_a = torch.tensor(tmp_a, dtype=torch.int32)
        arrays = tmp_a.view((1, 16, 16))
        file_name = r'E:\ddos\ddos{}.pt'.format(flowId)
        if i == 1:
            flows.append(flow)
            flows_time.append(packet.time)
            torch.save(arrays, file_name)
            print('完成flow', flowId)
            flowId = flowId + 1
            continue
        isUsed = 0

        for j in range(len(flows) - 1, -1, -1):

            if (float(packet.time) - float(flows_time[j])) > 600:
                isTimeOut = 1
            # 第一条数据
            if isSyn == 1:
                flows.append(flow)
                flows_time.append(packet.time)
                torch.save(arrays, file_name)
                flowId = flowId + 1
                isUsed = 1
                break

            # 5元组一样，分两种情况
            if operator.eq(flows[j][:5], flow[:5]) or operator.eq(flows[j][:5], re_flow[:5]):
                # 没有超时
                if isTimeOut != 1:
                    isUsed = 1
                    record_name = r'E:\ddos\ddos{}.pt'.format(j)
                    record_flow = torch.load(record_name)
                    # 超级数据包
                    flows[j][5] += 1
                    if record_flow.size(0) > 32:
                        break
                    merge_flow = torch.cat((record_flow, arrays), 0)
                    torch.save(merge_flow, record_name)
                    # print('flow %d 合并 flow %d' % (i, j))
                    break
                # 超时了
                else:
                    flows.append(flow)
                    flows_time.append(packet.time)
                    torch.save(arrays, file_name)
                    flowId = flowId + 1
                    isUsed = 1
                    break

        if isUsed == 0:
            flows.append(flow)
            flows_time.append(packet.time)
            torch.save(arrays, file_name)
            flowId = flowId + 1

        if sequence != 2 and (i + 1) % 5000 == 0:
            print('%d %d%% (%s)  %s' % (flowId + 1, 100 * (flowId + 1) / 4000, timeSince(start), flow))

        if (flowId + 1) % 2000 == 0:
            print('%d %d%% (%s)  %s' % (flowId + 1, 100 * (flowId + 1) / 100001, timeSince(start), flows[len(flows) - 1]))

        if flowId > 100001:
            break

df = pd.DataFrame(flows, columns=['Protocol', 'SourcePort', 'DestinationPort', 'SourceIP', 'DestinationIP', 'isGiant'])
df['Time'] = flows_time
df.to_csv('flows_benign.csv')


def getSpace():
    for i in range(180001):
        record_name = r'E:\ddos\ddos{}.pt'.format(i)
        # 归一化
        record_flow = torch.load(record_name) / 255.0
        space_tensor = torch.zeros(784)

        size = 0
        for k in range(record_flow.size(0)):
            x = torch.flatten(record_flow[k], 1)
            for j in range(x.size(1)):
                if x[0][j] != 0 and size < 784:
                    space_tensor[size] = x[0][j]
                    size = size + 1
        space_name = r'E:\space2\space{}.pt'.format(i)
        torch.save(space_tensor, space_name)
        if (i + 1) % 1000 == 0:
            print('%d %d%% (%s) ' % (i + 1, 100 * (i + 1) / 180001, timeSince(start),))


print('读取结束......')


# 切割tcp流：五元组(SourceIP, DestinationIP, SourcePort, DestinationPort, Protocol)
# attack = 18.219.211.138
# --------------------------------------------------------
# 目的mac地址6|源mac地址6|类型2|数据         46-1500|FCS4    |
# --------------------------------------------------------
# 版本|首部长度|    区分服务1     |总长度2                    |
#        标识 2                |标志3bit|片偏移17           |
# 生存时间1      |协议1         |首部校验和2                 |
#                          源地址4                         |
#                          目的地址4                       |
#                     可选字段(长度可变) 填充                |
# --------------------------------------------------------
# 以太网数据帧格式：目的mac地址6，源mac地址6，协议类型2，数据（14字节）
# ip数据格式（20），在源IP地址4，目的IP地址4，之前有12字节，因此混淆这两个数据需要在[27,34]之间填0
# 无论tcp（20）数据包还是udp（8）数据包2，源端口2，目的端口都紧随其后，因此最后是在[27,38]之间填0

# victim BOTNET
# -A "2018-03-02 22:11:00"  -B "2018-03-02 23:34:00"
# -A "2018-03-03 02:24:00"  -B "2018-03-03 03:55:00"
# 选取了4个文件
# capEC2AMAZ-O4EL3NG-172.31.69.23
# capEC2AMAZ-O4EL3NG-172.31.69.8
# capEC2AMAZ-O4EL3NG-172.31.69.12
# capEC2AMAZ-O4EL3NG-172.31.69.30
# DDOS ------------------
# editcap -A "2018-02-21 22:09:00"  -B "2018-02-21 22:43:00"
# editcap -A "2018-02-22 02:05:00"  -B "2018-02-22 03:05:00"