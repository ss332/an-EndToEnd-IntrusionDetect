import numpy as np
import matplotlib.pyplot as plt
from scapy import all
import time

cap_file = open(r'E:\Amazon\origin\pcap\capEC2AMAZ-O4EL3NG-172.31.69.23', 'rb')

cap_img = cap_file.read(800)

print(cap_img)

tmp_a = []
for byte in cap_img:
    tmp_a.append(byte)

tmp_a = np.array(tmp_a, dtype=np.uint8)
arrays = tmp_a.reshape((20, 40), order='C')
print(arrays)
plt.imshow(arrays)
plt.show()

# 切割tcp流：六元组（FlowId,SourceIP, DestinationIP, SourcePort, DestinationPort, Protocol)
# E:\Amazon\origin\pcap\UCAP172.31.69.25 print (time.mktime(time.strptime(a,"%Y-%m-%d %H:%M:%S"))) editcap -A
# "2018-03-02 22:11:00"  -B "2018-03-03 08:34:00"  E:\Amazon\origin\pcap\capEC2AMAZ-O4EL3NG-172.31.69.23
# E:\Amazon\DDOS-LOIC-UDP.pcap
# attack = 18.219.211.138
packets = all.PcapReader(r'E:\Amazon\DDOS-LOIC-UDP.pcap')
print(packets.read_packet().time)
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(packets.read_packet().time)))
print(packets.read_packet().show())

flows = []
packets = all.rdpcap(r'E:\Amazon\DDOS-LOIC-UDP.pcap')
i = 0
for packet in packets:
    if packet.haslayer('TCP'):
        flow = [i, packet['IP'].src, packet['IP'].dst, packet['TCP'].sport,packet['TCP'].sport,packet['IP'].proto]
