import numpy as np
import pandas as pd
import time
import re
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scapy import all

# -A "2018-03-02 22:11:00"  -B "2018-03-02 23:34:00"
# -A "2018-03-03 02:24:00"  -B "2018-03-03 03:55:00"

# 将格式字符串转换为时间戳
a_start = "Mar 02 22:11:00 2018"
a_end = "Mar 02 23:34:00 2018"
c_start = "Mar 03 02:24:00 2018"
c_end = "Mar 03 03:55:00 2018"

print(time.mktime(time.strptime(a_start, "%b %d %H:%M:%S %Y")))
print(time.mktime(time.strptime(a_end, "%b %d %H:%M:%S %Y")))
print(time.mktime(time.strptime(c_start, "%b %d %H:%M:%S %Y")))
print(time.mktime(time.strptime(c_end, "%b %d %H:%M:%S %Y")))

time1=[1519999860.0,1520004840.0]
time2=[1520015040.0,1520020500.0]
print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(1519999860.0)))