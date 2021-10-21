import pandas as pd
from sklearn.utils import shuffle

flows = pd.read_csv('./resources/flows_bot.csv')
print(flows.head())

labels = []
ips = ['172.31.69.23', '172.31.69.8', '172.31.69.30','172.31.69.17','172.31.69.12','172.31.69.29']


time1 = [1519999860.0, 1520004840.0]
time2 = [1520015040.0, 1520020500.0]


attacker_ip = '18.219.211.138'
port=8080

for i in range(100002):
    match_ip = 0
    match_time = 0
    dst_ip = flows['DestinationIP'][i]
    protocol = flows['Protocol'][i]
    dst_port = flows['DestinationPort'][i]

    if time1[0] <= flows['Time'][i] <= time1[1] and protocol == 6:
        match_time = 1
    for ip in ips:
        if flows['SourceIP'][i] == ip:
            match_ip = 1

    if match_ip == 1 and dst_ip == attacker_ip and dst_port == port:
        labels.append(1)

    else:
        labels.append(0)

flows['label'] = labels

flows.to_csv('flows1.csv')
print(flows[flows['label'] == 1].shape)
