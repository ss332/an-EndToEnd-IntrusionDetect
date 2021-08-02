import numpy as np
import pandas as pd
import time
import re
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
a=np.zeros((28,28))
a=a.reshape((1,28,28))
c=np.zeros((28,28))
c=a.reshape((1,28,28))
d=np.concatenate((a,c),axis=0)
print(d,d.shape)