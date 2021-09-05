import pandas as pd
import torch.nn as nn
import torch
import time
from sklearn.utils import shuffle
import numpy as np
import math
import torch.optim as optim
import warnings

warnings.filterwarnings("ignore")

# flows数据记录表，表明每一个流，具体数据从E:\bot\FlowId{}.npy根据id取得
flows = pd.read_csv('flows1.csv')
flows = shuffle(flows)
# 标签集
labelSet = flows["label"]


# 记录时间
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# 模型，从卷积网络到神经网络
class Cnn2Rnn(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Cnn2Rnn, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(800, 256)

        self.gru = nn.GRU(256, hidden_size)
        self.out = nn.Linear(256, output_size)

        self.dropout = nn.Dropout(0.2)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        # # cnn网络
        # # 1*14*14 7
        x = self.pool(self.relu(self.conv1(x)))
        # # 1*5*5
        x = self.relu(self.conv2(x))

        # 1*800
        x = torch.flatten(x, 1)  # 除去批次维度，其他展平 [N,256]
        # 1*64
        x = self.relu(self.fc1(x))

        # 进入rnn网络
        # x= 1*1*64
        x = x.unsqueeze(0)
        # out= 1*1*256,hidden=1*1*256
        output, hidden = self.gru(x, hidden)
        # out=1*1*2
        output = self.out(output)
        # drop=0.2
        output = self.dropout(output)
        # out=1*1*2
        output = output.squeeze(0)
        output = self.logSoftmax(output)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


# 参数设置，rnn隐藏层为256
hidden_size = 256
# 实例化模型
c2rnn = Cnn2Rnn(hidden_size, 2)
# 损失函数，负对数似然
criterion = nn.NLLLoss()
# 优化函数
optimizer = torch.optim.Adam(c2rnn.parameters(), lr=0.001)

# 平均模型，记录后几个纪元更新参数的平均值，取其作为模型的最终参数
swa_model = optim.swa_utils.AveragedModel(c2rnn)
# 学习率衰减计划，按cos曲线衰减，最大周期为10
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6)
# swa学习率衰减计划,an
swa_scheduler = optim.swa_utils.SWALR(optimizer, swa_lr=0.0001)


# inputtensor = n*28*28,catergory_tensor=[1]or[0]
# 获取矩阵化的数据包与标签
def getInputAndTargerTensor(id):
    id = flows['id'].iloc[id]
    record_name = r'E:\bot\FlowId{}.npy'.format(id)
    # 对其归一化/255.0
    record_flow = np.load(record_name) / 255.0
    input_Tensor = torch.from_numpy(record_flow)
    input_Tensor = input_Tensor.to(torch.float32)
    target_tensor = labelSet[id]

    return input_Tensor, target_tensor


def train(input_tensor, target_tensor):
    global output
    # 传入是参数int,需要转化为[int]
    target_tensor = torch.tensor(target_tensor).unsqueeze(0)
    hidden = c2rnn.initHidden()

    # 参数初始化
    optimizer.zero_grad()
    loss = 0
    # input_tensor 是n*28*28,需要输入1*28*28
    for i in range(input_tensor.size(0)):
        output, hidden = c2rnn(input_tensor[i].unsqueeze(0).unsqueeze(0), hidden)

        l = criterion(output, target_tensor)
        loss += l
    # 梯度后向传播
    loss.backward()
    # 更新参数
    optimizer.step()

    return output, loss.item() / input_tensor.size(0)


n_iters = 139253
print_every = 2000
plot_every = 500
# 记录损失曲线
all_losses = []
start = time.time()


# 训练，循环训练集10个纪元
def trainIter():
    print('Training start!')
    for epoch in range(2):  # loop over the dataset multiple times

        plot_loss = 0.0
        print_loss = 0.0
        for iter in range(n_iters + 1):

            output, loss = train(*getInputAndTargerTensor(iter))
            plot_loss += loss
            print_loss += loss

            if (iter + 1) % print_every == 0:  # print every 5000 mini-batches
                print('Epoch[%d,%5d, %d%%] -%s /step  -loss: %.4f ' % (
                    epoch + 1, iter + 1, (iter + 1) / (n_iters + 1) * 100, timeSince(start, iter / n_iters),
                    print_loss / print_every))
                print_loss = 0

            if iter % plot_every == 0:
                all_losses.append((plot_loss / plot_every))
                plot_loss = 0
        if epoch > 2:
            swa_model.update_parameters(c2rnn)
            swa_scheduler.step()
        else:
            scheduler.step()

    print('Training Finished')
    # 没有batch  batch normalization，可以不用
    # optim.swa_utils.update_bn(trainloader, swa_model)
    with torch.no_grad():
        for i in range(0, 139254):
            input_tensor, target_tensor = getInputAndTargerTensor(i)
            hidden = c2rnn.initHidden()
            # 这里输入的是N*D*W*H(批次，维度，宽，高）
            for j in range(input_tensor.size(0)):
                output, hidden = swa_model(input_tensor[j].unsqueeze(0).unsqueeze(0), hidden)


# 验证
# 用测试集验证，求出混淆矩阵

def evaluate():
    total = 0
    tp, fp, tn, fn = 0.000001, 0.000001, 0.000001, 0.000001

    with torch.no_grad():
        for i in range(139254, 198934):
            input_tensor, target_tensor = getInputAndTargerTensor(i)
            global output

            hidden = c2rnn.initHidden()
            # 这里输入的是N*D*W*H(批次，维度，宽，高）
            for j in range(input_tensor.size(0)):
                output, hidden = swa_model(input_tensor[j].unsqueeze(0).unsqueeze(0), hidden)

            top_n, top_i = output.topk(1)
            predicted = top_i[0].item()

            total = total + 1

            if target_tensor == 1 and predicted == 1:
                tp += 1
            if target_tensor == 0 and predicted == 0:
                tn += 1
            if target_tensor == 1 and predicted == 0:
                fn += 1
            if target_tensor == 0 and predicted == 1:
                fp += 1

    accuracy = 100 * (tp + tn) / total
    recall = 100 * tp / (tp + fn)
    precision = 100 * tp / (tp + fp)
    f1_score = 2 * precision * recall / (precision + recall)

    print('测试集共计%d ,比例为0.3' % (185010))
    print('Accuracy : %d %% (%d)' % (accuracy, tp + tn))
    print('recall : %d %% (%d)' % (recall, tp))
    print('precision : %d %% (%d)' % (precision, tp))
    print('f1Score : %d %% ' % (f1_score))


# 开始！
trainIter()
torch.save(swa_model.state_dict(), 'swa_model_weights.pth')
# swa_model.load_state_dict(torch.load('swa_model_weights.pth'))
evaluate()
