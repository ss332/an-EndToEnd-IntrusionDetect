import pandas as pd
from sklearn.utils import shuffle

time1 = [1518618720.0, 1518624540.0, 1518631260.0, 1518636660.0]
time2 = [1518701160.0, 1518703740.0, 1518706740.0, 1518709200.0]
time3 = [1518790320.0, 1518793680.0, 1518803100.0, 1518805140.0]
time4 = [1519135920.0, 1519139820.0]
time5 = [1519222140.0, 1519224180.0, 1519236300.0, 1519239900.0]
time6 = [1519309020.0, 1519313040.0, 1519321800.0, 1519324140.0, 1519330500.0, 1519331340.0]
time7 = [1519829400.0, 1519833900.0, 1519839720.0, 1519843200.0]
time8 = [1519999860.0, 1520004840.0, 1520015040.0, 1520020500.0]

s_ip1 = ['18.221.219.4', '13.58.98.64']
s_ip2 = ['18.219.211.138', '18.217.165.70']
s_ip3 = ['13.59.126.31', '18.219.193.20']
s_ip4 = ['18.218.115.60', '18.219.9.1', '18.219.32.43', '18.218.55.126', '52.14.136.135', '18.219.5.43',
         '18.216.200.189',
         '18.218.229.235', '18.218.11.51', '18.216.24.42']
s_ip5 = ['18.218.115.60', '18.219.9.1', '18.219.32.43', '18.218.55.126', '52.14.136.135', '18.219.5.43',
         '18.216.200.189',
         '18.218.229.235', '18.218.11.51', '18.216.24.42']
s_ip6 = ['18.218.115.60']
s_ip7 = ['13.58.225.34']
s_ip8 = ['18.219.211.138']

d_ip1 = ['172.31.69.25']
d_ip2 = ['172.31.69.25']
d_ip3 = ['172.31.69.25']
d_ip4 = ['172.31.69.25']
d_ip5 = ['172.31.69.28']
d_ip6 = ['172.31.69.28']
d_ip7 = ['172.31.69.24']
d_ip8 = ['172.31.69.23', '172.31.69.17', '172.31.69.14', '172.31.69.12', '172.31.69.10', '172.31.69.8',
         '172.31.69.6', '172.31.69.26', '172.31.69.29', '172.31.69.30']

pro1 = [21, 6, 22, 6]
pro2 = [80, 6, 80, 6]
pro3 = [21, 6, 80, 6]
pro4 = [80, 6]
pro5 = [80, 17, 80, 6]
pro6 = [80, 6, 80, 6, 80, 6]
pro7 = [6]
pro8 = [8080, 6]
s_ip = [s_ip1, s_ip2, s_ip3, s_ip4, s_ip5, s_ip6, s_ip7, s_ip8]
d_ip = [d_ip1, d_ip2, d_ip3, d_ip4, d_ip5, d_ip6, d_ip7, d_ip8]


def tiks(sequence, c1, c2):
    flows = pd.read_csv(f'./files/file_{sequence}.csv')
    flows['file'] = 'files{}'.format(sequence)
    print(flows.head())

    labels = []
    for i in range(100000):
        time = time2
        match_s_ip = 0
        match_d_ip = 0
        match_time = 0

        source_ip = flows['sourceIP'][i]
        dst_ip = flows['destinationIP'][i]
        protocol = flows['protocol'][i]
        dst_port = flows['destinationPort'][i]
        packet_nums = flows['packets'][i]

        if time[0] <= flows['time'][i] <= time[1] and dst_port == 80 and protocol == 6 and source_ip == s_ip2[0]:
            match_time = 1
        if time[2] <= flows['time'][i] <= time[3] and dst_port == 80 and protocol == 6 and source_ip == s_ip2[1]:
            match_time = 2

        # for ip in s_ip[sequence - 1]:
        #     if source_ip == ip:
        #         match_s_ip = 1

        for ip in d_ip[sequence - 1]:
            if dst_ip == ip:
                match_d_ip = 1

        if match_s_ip == 0 and match_d_ip == 1:
            if match_time == 1:
                labels.append(c1)
            elif match_time == 2:
                labels.append(c2)
            else:
                labels.append(0)
        else:
            labels.append(0)

    flows['label'] = labels
    flows = flows.drop('id', axis=1)

    flows.to_csv('./files/f_process{}.csv'.format(sequence))
    print(flows[flows['label'] == c1].shape)
    print(flows[flows['label'] == c2].shape)


tiks(2,3,4)
flows = pd.read_csv('./files/f_process2.csv')
print(flows[flows['label'] == 0].shape)
print(flows[flows['label'] == 3].shape)
print(flows[flows['label'] == 4].shape)
print(flows[flows['label'] == 12].shape)

