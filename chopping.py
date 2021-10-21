import pandas as pd
from sklearn.utils import shuffle

flows = pd.read_csv('./resources/flows_ddos.csv')
print(flows.head())

labels = []

ips2 = ['18.218.115.60', '18.219.9.1', '18.219.32.43', '18.218.55.126', '52.14.136.135', '18.219.5.43',
        '18.216.200.189',
        '18.218.229.235', '18.218.11.51', '18.216.24.42']

time3 = [1519222140.0, 1519224180.0]
time4 = [1519236300.0, 1519239900.0]

victim_ip = '172.31.69.28'
victim_port = 80

for i in range(100001):
    match_ip = 0
    match_time = 0
    dst_ip = flows['DestinationIP'][i]
    protocol = flows['Protocol'][i]
    dst_port = flows['DestinationPort'][i]
    packet_nums=flows['isGiant'][i]

    if time3[0] <= flows['Time'][i] <= time3[1] and protocol == 17:
        match_time = 1
    if time4[0] <= flows['Time'][i] <= time4[1] and protocol == 6:
        match_time = 2
    for ip in ips2:
        if flows['SourceIP'][i] == ip:
            match_ip = 1

    if match_ip == 1 and dst_ip == victim_ip and dst_port == victim_port:
        if match_time == 1 and packet_nums > 1000:
            labels.append(1)
        elif match_time == 2:
            labels.append(2)
        else:
            labels.append(0)
    else:
        labels.append(0)

flows['label'] = labels
flows.drop('id', axis=1, inplace=True)
flows.to_csv('flows2.csv')
print(flows[flows['label'] == 2].shape)
flows = shuffle(flows)
print(flows.head(8))
