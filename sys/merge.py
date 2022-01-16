import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# merge_flows=pd.read_csv('../files/f_process1.csv')
# for i in range(7):
#     flows=pd.read_csv('../files/f_process{}.csv'.format(i+2))
#     merge_flows= pd.concat([merge_flows,flows], axis=0)
#
# merge_flows.to_csv('all_flows.txt',index=0)

label=[]
flows=pd.read_csv('all_flows.txt')
for i in range(800000):
    if  flows['label'][i]==0:
        label.append(0)
    else:
        label.append(1)

flows = flows.drop('label', axis=1)
flows['label']=label
flows.to_csv('two_flows.txt',index=0)