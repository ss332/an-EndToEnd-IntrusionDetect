# 记录时间
import math
import time
import torch
import pandas as pd

answer_words = ['Benign', 'FTP-BruteForce', 'SSH-Bruteforce', 'DoS attacks-GoldenEye', 'DoS attacks-Slowloris',
                'DoS attacks-Hulk', 'DoS attacks-SlowHTTPTest', 'DDoS attacks-LOIC-HTTP', 'DDOS attack-LOIC-UDP',
                'DDOS attack-HOIC', 'Brute Force -Web', 'Brute Force -XSS', 'SQL Injection',
                'Infilteration',
                'Bot']


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(all,since, percent):
    now = time.time()
    s1 = now - all
    s2 = now - since
    es = s2/ (percent)
    rs = es - s2
    return '%s (- %s)' % (asMinutes(s1), asMinutes(rs))


def class_name(file_id, id):
    df = pd.read_csv('../files/f_process{}.csv'.format(file_id))
    label = df['label'].iloc[id]
    label=answer_words[label]
    return label

# for i in range(14):
#     print(i+1)
#     a=torch.load('p{}.pt'.format(i + 1))
#     s,rank=a.sort(0,descending=True)
#     rank=rank+1
#     print(a)
#     print(rank)
