import pandas as pd
from sklearn.utils import shuffle

flows = pd.read_csv('./resources/flows.csv')
print(flows.head())
print(flows['id'].max())
labels = []

for i in range(flows['id'].max() + 1):
    if flows['SourceIP'][i] == '172.31.69.23' and flows['DestinationIP'][i] == '18.219.211.138' and flows['Protocol'][i] == 6:
        labels.append(1)
    else:
        labels.append(0)

flows['label'] = labels

print(flows[flows['label'] == 1].shape)
flows = shuffle(flows)
print(flows.head(8))
