import pandas as pd
from sklearn.utils import shuffle

flows = pd.read_csv('./resources/flows.csv')
print(flows.head())
print(flows['id'].max())

labels = []
ips = ['172.31.69.23', '172.31.69.8', '172.31.69.30']
time1 = [1519999860.0, 1520004840.0]
time2 = [1520015040.0, 1520020500.0]
attacker_ip = '18.219.211.138'
attacker_port = 8080

for i in range(flows['id'].max() + 1):
    match_ip = 0
    match_time = 0
    dst = flows['DestinationIP'][i]
    protocol = flows['Protocol'][i]
    dst_port = flows['DestinationPort'][i]

    if time1[0] <= flows['Time'][i] <= time1[1] or time2[0] <= flows['Time'][i] <= time2[1]:
        match_time = 1

    for ip in ips:
        if flows['SourceIP'][i] == ip:
            match_ip = 1

    if match_ip == 1 and match_time == 1 and dst == attacker_ip and protocol == 6 and dst_port == attacker_port:
        labels.append(1)
    else:
        labels.append(0)

flows['label'] = labels


flows.to_csv('flows1.csv')
print(flows[flows['label'] == 1].shape)
flows = shuffle(flows)
print(flows.head(8))



