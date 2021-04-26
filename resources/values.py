from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score)
# 评估指标
""""
model_name :模型名字
y_true,y_pred：对应真实标签和模型预测 
times：模型训练时间
"""


def printMetrics(model_name, y_true, y_pred, times):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    print('{}指标表现：'.format(model_name))
    print('训练用时{}'.format(times))
    print('accuracy:{}'.format(accuracy))
    print('precision:{},\t all precision:{}'.format(precision, precision_score(y_true, y_pred, average='macro')))
    print('recall:{},\t all recall:{}'.format(recall, recall_score(y_true, y_pred, average='macro')))
    print('f1_score:{},\t all recall:{}'.format(f1, f1_score(y_true, y_pred, average='macro')))
    print('------------------------------------------------------------')


col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
             "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
             "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
             "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
             "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
             "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
             "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
             "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
             "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

nsl_label_replace = {'normal': 0, 'neptune': 1, 'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,
                      'mailbomb': 1, 'apache2': 1,
                      'processtable': 1, 'udpstorm': 1, 'worm': 1,
                      'ipsweep': 2, 'nmap': 2, 'portsweep': 2, 'satan': 2, 'mscan': 2, 'saint': 2
    , 'ftp_write': 3, 'guess_passwd': 3, 'imap': 3, 'multihop': 3, 'phf': 3, 'spy': 3, 'warezclient': 3,
                      'warezmaster': 3, 'sendmail': 3, 'named': 3, 'snmpgetattack': 3, 'snmpguess': 3, 'xlock': 3,
                      'xsnoop': 3,
                      'httptunnel': 3,
                      'buffer_overflow': 4, 'loadmodule': 4, 'perl': 4, 'rootkit': 4, 'ps': 4, 'sqlattack': 4,
                      'xterm': 4}


def getNslColNames():
    return col_names


def getNslLabelReplace():
    return nsl_label_replace
