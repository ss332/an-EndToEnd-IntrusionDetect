import numpy as np
import matplotlib.pyplot as plt
from scapy import all

cap_file = open(r'E:\Amazon\origin\pcap\capDESKTOP-AN3U28N-172.31.64.17', 'rb')

cap_img = cap_file.read(800)

print(cap_img)

tmp_a = []
for byte in cap_img:
    tmp_a.append(byte)

tmp_a = np.array(tmp_a, dtype=np.uint8)
arrays=tmp_a.reshape((20,40),order='C')
print(arrays)
plt.imshow(arrays)
plt.show()

# 切割tcp流：
packets = all.rdpcap(r'E:\Amazon\origin\pcap\capDESKTOP-AN3U28N-172.31.64.26')
print(packets[12].show())

