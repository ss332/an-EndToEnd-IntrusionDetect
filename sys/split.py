from scapy import all
import time
import operator
import re
import pandas as pd
import math
import torch

""""
切割tcp流：五元组(SourceIP, DestinationIP, SourcePort, DestinationPort, Protocol)
--------------------------------------------------------
目的mac地址6|源mac地址6|类型2|数据         46-1500|FCS4    |
--------------------------------------------------------
版本|首部长度|    区分服务1     |总长度2                    |
       标识 2                |标志3bit|片偏移17           |
生存时间1      |协议1         |首部校验和2                 | 
                         源地址4                         |
                         目的地址4                       |
                    可选字段(长度可变) 填充                |
--------------------------------------------------------
以太网数据帧格式：目的mac地址6，源mac地址6，协议类型2，数据（14字节）
ip数据格式（20），在源IP地址4，目的IP地址4，之前有12字节，因此混淆这两个数据需要在[27,34]之间填0
无论tcp（20）数据包还是udp（8）数据包2，源端口2，目的端口都紧随其后，因此最后是在[27,38]之间填0
aws s3 ls --no-sign-request "s3://cse-cic-ids2018/Original Network Traffic and Log data/Friday-23-02-2018/pcap.zip" --recursive --human-readable --summarize
aws s3 cp --no-sign-request "s3://cse-cic-ids2018/"
"""
# 1 5 6 7 8
files1 = [r'H:\ids2018\wed-14-02\UCAP172.31.69.25',
          r'H:\ids2018\wed-14-02\capEC2AMAZ-O4EL3NG-172.31.69.24',
          r'H:\ids2018\wed-14-02\capEC2AMAZ-O4EL3NG-172.31.69.23',
          r'H:\ids2018\wed-14-02\capEC2AMAZ-O4EL3NG-172.31.69.28']  # ftp,ssh bruteforce
files2 = [r'H:\ids2018\thurs-15-02\UCAP172.31.69.25',
          r'H:\ids2018\thurs-15-02\capEC2AMAZ-O4EL3NG-172.31.69.17',
          r'H:\ids2018\thurs-15-02\capEC2AMAZ-O4EL3NG-172.31.69.8',
          r'H:\ids2018\thurs-15-02\capEC2AMAZ-O4EL3NG-172.31.69.12',
          r'H:\ids2018\thurs-15-02\capEC2AMAZ-O4EL3NG-172.31.69.29',
          r'H:\ids2018\thurs-15-02\capEC2AMAZ-O4EL3NG-172.31.69.30']  # dos-goldenEye,slowloris
files3 = [r'H:\ids2018\fri-16-02\UCAP172.31.69.25-part1.pcap',  # dos slowhttptest,hulk
          r'H:\ids2018\fri-16-02\UCAP172.31.69.25-part2.pcap',
          r'H:\ids2018\fri-16-02\capEC2AMAZ-O4EL3NG-172.31.69.23',
          r'H:\ids2018\fri-16-02\capEC2AMAZ-O4EL3NG-172.31.69.24',
          r'H:\ids2018\fri-16-02\capEC2AMAZ-O4EL3NG-172.31.69.26',
          r'H:\ids2018\fri-16-02\capEC2AMAZ-O4EL3NG-172.31.69.28',
          r'H:\ids2018\fri-16-02\capEC2AMAZ-O4EL3NG-172.31.69.29',
          r'H:\ids2018\fri-16-02\capEC2AMAZ-O4EL3NG-172.31.69.30',
          ]
files4 = [r'E:\Amazon\tuesday-20-02\UCAP172.31.69.25',
          r'E:\Amazon\tuesday-20-02\capEC2AMAZ-O4EL3NG-172.31.69.23',
          r'E:\Amazon\tuesday-20-02\capEC2AMAZ-O4EL3NG-172.31.69.24',
          r'E:\Amazon\tuesday-20-02\capEC2AMAZ-O4EL3NG-172.31.69.28',
          r'E:\Amazon\tuesday-20-02\capEC2AMAZ-O4EL3NG-172.31.69.29']  # dos loic-http
files5 = [r'E:\Amazon\wednes-21-02\UCAP172.31.69.28-part1',  # dos loic-udp,hoic 耗时长
          r'E:\Amazon\wednes-21-02\UCAP172.31.69.28-part2']
files6 = [r'H:\ids2018\Thurs-22-02\UCAP172.31.69.28',
          r'H:\ids2018\Thurs-22-02\UCAP172.31.69.21',
          r'H:\ids2018\Thurs-22-02\UCAP172.31.69.22',
          r'H:\ids2018\Thurs-22-02\UCAP172.31.69.25',
          r'H:\ids2018\Thurs-22-02\capEC2AMAZ-O4EL3NG-172.31.69.23',
          r'H:\ids2018\Thurs-22-02\capEC2AMAZ-O4EL3NG-172.31.69.17',
          r'H:\ids2018\Thurs-22-02\capEC2AMAZ-O4EL3NG-172.31.69.14',
          r'H:\ids2018\Thurs-22-02\capEC2AMAZ-O4EL3NG-172.31.69.10',
          r'H:\ids2018\Thurs-22-02\capEC2AMAZ-O4EL3NG-172.31.69.8',
          r'H:\ids2018\Thurs-22-02\capEC2AMAZ-O4EL3NG-172.31.69.6',
          r'H:\ids2018\Thurs-22-02\capEC2AMAZ-O4EL3NG-172.31.69.12',
          r'H:\ids2018\Thurs-22-02\capEC2AMAZ-O4EL3NG-172.31.69.26',
          r'H:\ids2018\Thurs-22-02\capEC2AMAZ-O4EL3NG-172.31.69.29',
          r'H:\ids2018\Thurs-22-02\capEC2AMAZ-O4EL3NG-172.31.69.30'
          ]  # web,xss bruteforce,sql injection

files7 = [r'D:\ids2018\wed-28-02\capEC2AMAZ-O4EL3NG-172.31.69.24-part2',
          r'D:\ids2018\wed-28-02\capEC2AMAZ-O4EL3NG-172.31.69.17',
          r'D:\ids2018\wed-28-02\capEC2AMAZ-O4EL3NG-172.31.69.23',
          r'D:\ids2018\wed-28-02\capEC2AMAZ-O4EL3NG-172.31.69.26',
          r'D:\ids2018\wed-28-02\capEC2AMAZ-O4EL3NG-172.31.69.30',
          r'D:\ids2018\wed-28-02\capEC2AMAZ-O4EL3NG-172.31.69.10',
          r'D:\ids2018\wed-28-02\capEC2AMAZ-O4EL3NG-172.31.69.12',
          r'D:\ids2018\wed-28-02\capEC2AMAZ-O4EL3NG-172.31.69.14'
          ]  # infiltration
files8 = [r'E:\Amazon\friday-02-03\capEC2AMAZ-O4EL3NG-172.31.69.23',  # bot
          r'E:\Amazon\friday-02-03\capEC2AMAZ-O4EL3NG-172.31.69.17',
          r'E:\Amazon\friday-02-03\capEC2AMAZ-O4EL3NG-172.31.69.14',
          r'E:\Amazon\friday-02-03\capEC2AMAZ-O4EL3NG-172.31.69.12',
          r'E:\Amazon\friday-02-03\capEC2AMAZ-O4EL3NG-172.31.69.29',
          r'E:\Amazon\friday-02-03\capEC2AMAZ-O4EL3NG-172.31.69.30']

sesseion_num = 100002
# 缓存的大小
cahe_length = 200
# 统计io次数
io_count = 0
# id持久度

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def update_cache(list, arrays, id, seq):
    if id == sesseion_num:
        a = 1
        for session in list:
            file_name = r'C:\sessions\files{}\session{}.pt'.format(seq, id - a)
            torch.save(session, file_name)
            a = a+1

    elif len(list) < cahe_length:
        list.insert(0, arrays)
    else:
        file_name = r'C:\sessions\files{}\session{}.pt'.format(seq, id - cahe_length)

        torch.save(list[len(list) - 1], file_name)
        # 老的保存，新的插入
        del list[len(list) - 1]
        list.insert(0, arrays)


def check_cache(cache_sessions):
    cool = 0
    for session in cache_sessions:
        if session.size(0) < 32:
            cool = 1

    return cool


def split(files, seq, count_io=0):
    # 提高速度，保有最近n个的session矩阵,减少IO读取
    cache_sessions = []
    flows = []
    flows_time = []
    flow_id = 0
    sequence = 0

    start = time.time()
    print('处理文件{}读取开始......,'.format(seq))

    for file in files:
        if flow_id == sesseion_num:
            break

        sequence += 1
        packets = all.PcapReader(file)
        print('file', sequence)
        hasnext = True
        i = 0

        while hasnext:
            try:
                packet = packets.read_packet()
            except:
                packets.close()
                hasnext = False
                continue
            i = i + 1
            if i > 1000001:
                i = 2
            isSyn = 0
            isFin = 0
            #  特殊控制（files1 :1518631260.0）  （files3: 1518803100.0,id>28600）  （files5：1519236300 1519222140）
            # and float(packet.time) - 1518803100 < 0
            # if sequence==1 and flow_id > 30000 and i>1000000:
            #     if i % 50000 == 0:
            #         print('skip', i)
            #     break

            if packet.haslayer('IP'):
                if packet.haslayer('TCP'):
                    flags = str(packet['TCP'].flags)
                    # # tcp三次握手第一个syn包，ack=0，因此flags里不应有A
                    # if re.search('S', flags) is not None and re.search('A', flags) is None:
                    #     isSyn = 1
                    if re.search('F', flags) is not None:
                        isFin = 1

                    flow = [packet['IP'].proto, packet['TCP'].sport, packet['TCP'].dport, packet['IP'].src,
                            packet['IP'].dst, 1, 0, 0, 0]
                    re_flow = [packet['IP'].proto, packet['TCP'].dport, packet['TCP'].sport, packet['IP'].dst,
                               packet['IP'].src, 1, 0, 0, 0]

                elif packet.haslayer('UDP'):

                    flow = [packet['IP'].proto, packet['UDP'].sport, packet['UDP'].dport, packet['IP'].src,
                            packet['IP'].dst, 1, 0, 0, 0]
                    re_flow = [packet['IP'].proto, packet['UDP'].dport, packet['UDP'].sport, packet['IP'].dst,
                               packet['IP'].src, 1, 0, 0, 0]
            else:
                continue

            tmp_a = torch.zeros(320)
            s = 0
            data = bytes(packet)
            flow[8] = len(data)

            for byte in data:
                tmp_a[s] = byte
                s = s + 1
                if s == 320:
                    break
            # 归一化
            arrays = tmp_a.view((1, 320)) / 255.0

            if i == 1:
                flows.append(flow)
                flows_time.append(packet.time)
                update_cache(cache_sessions, arrays, flow_id, seq)
                print('完成flow', flow_id)
                flow_id = flow_id + 1
                continue
            isUsed = 0
            length = len(flows)
            for j in range(length - 1, -1, -1):

                # 5元组一样，分两种情况
                if operator.eq(flows[j][:5], flow[:5]) or operator.eq(flows[j][:5], re_flow[:5]):
                    duration = float(packet.time) - float(flows_time[j])
                    timeout = duration - flows[j][6]
                    # 没有超时
                    if timeout <= 600:
                        isUsed = 1
                        record_name = r'C:\sessions\files{}\session{}.pt'.format(seq, j)
                        flows[j][5] += 1
                        flows[j][6] = duration
                        if isFin == 1:
                            flows[j][7] = 1
                        flows[j][8] += flow[8]
                        if length - 1 >= j > length - (cahe_length + 1):
                            record_flow = cache_sessions[length - 1 - j]
                            if record_flow.size(0) > 32:
                                break
                            merge_flow = torch.cat((record_flow, arrays), 0)
                            cache_sessions[length - 1 - j] = merge_flow
                        else:
                            record_flow = torch.load(record_name)
                            count_io += 1
                            if record_flow.size(0) > 32:
                                break
                            merge_flow = torch.cat((record_flow, arrays), 0)

                            torch.save(merge_flow, record_name)
                        # 超级数据包
                        break
                    # 超时了
                    else:
                        flows.append(flow)
                        flows_time.append(packet.time)
                        flows[j][6] = duration
                        update_cache(cache_sessions, arrays, flow_id, seq)
                        count_io += 1
                        flow_id = flow_id + 1
                        isUsed = 1
                        break

            if isUsed == 0:
                flows.append(flow)
                flows_time.append(packet.time)
                update_cache(cache_sessions, arrays, flow_id, seq)
                count_io += 1
                flow_id = flow_id + 1
            # 主要针对数据包较多的情况，如ddos-udp
            if sequence == 1 and i % 5000 == 0:
                print('%d %d%% (%s)  %s %s %d' % (flow_id, flow_id / 40, timeSince(start), flow, packet.time, count_io))
            # 一般情况打印进度
            if flow_id % 2000 == 0:
                print('%d %d%% (%s)  %s' % (
                    flow_id, flow_id / 1000, timeSince(start), flows[len(flows) - 1]))
            # 只取10w个会话
            # if flow_id > 300 and sequence == 1:
            #     break
            if flow_id == sesseion_num:
                update_cache(cache_sessions, arrays, flow_id, seq)
                break

    df = pd.DataFrame(flows,
                      columns=['protocol', 'sourcePort', 'destinationPort', 'sourceIP', 'destinationIP', 'packets',
                               'duration', 'isFin', 'bytes'])

    df['mean_bytes'] = (df['bytes'] / df['packets'])
    df['time'] = flows_time

    df.to_csv(f'../files/file_{seq}.csv')
    print('读取结束......')


split(files8, 8)
