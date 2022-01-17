import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch
import time
from sklearn.utils import shuffle
import math
import torch.optim as optim
import warnings
import spacetime
import numpy as np

warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# flows数据记录表，表明每一个流，具体数据根据id取得
flows = pd.read_csv('all_flows.txt')
flows = shuffle(flows)
# 标签集
labelSet = flows["label"].to(device)
print(flows["label"].count())
set_size = 800000
train_size = 560000
test_size = 240000


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


def getInputAndTargerTensor(i):
    file = flows["file"].iloc[i]
    id = flows["id"].iloc[i]
    flow_name = r'D:\sessions\{}\session{}.pt'.format(file, id)

    # 归一化
    # n*320
    input_tensor = torch.load(flow_name).to(device)

    # 1*1024
    space_tensor = torch.zeros(1024,device=device)
    size = 0

    for k in range(input_tensor.size(0)):
        x = input_tensor[k]
        for j in range(x.size(0)):
            if x[j] != 0 and size < 1024:
                space_tensor[size] = x[j]
                size = size + 1

    # (b,c,w,h)
    space_tensor = space_tensor.view(1, 1, 32, 32)
    target_tensor = labelSet[id]

    return input_tensor, space_tensor, target_tensor


def train(input_tensor, space_tensor, target_tensor, rnn, space, time_optimizer, space_optimizer, criterion):
    rnn_hidden = rnn.initHidden()
    target_tensor = torch.tensor(target_tensor).unsqueeze(0)

    time_optimizer.zero_grad()
    space_optimizer.zero_grad()

    input_length = input_tensor.size(0)

    loss = 0
    rnn_output = torch.zeros((1, 64))
    for ei in range(input_length):
        rnn_output, rnn_hidden = rnn(
            input_tensor[ei], rnn_hidden)

    space_output = space(space_tensor, rnn_output)
    loss += criterion(space_output, target_tensor)

    loss.backward()
    time_optimizer.step()
    space_optimizer.step()

    return loss.item()


print_every = 2000
plot_every = 500
# 记录损失曲线
all_losses = []
start = time.time()

hidden_size = 320
timeRnn_model = spacetime.RNN(hidden_size).to(device)
spaceCnn_model = spacetime.Space().to(device)


# 训练，循环训练集10个纪元
def trainIter():
    print('Training start!')

    criterion = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(timeRnn_model.parameters(), lr=0.001)
    optimizer2 = torch.optim.Adam(spaceCnn_model.parameters(), lr=0.001)

    # 学习率衰减计划，按cos曲线衰减，最大周期为10
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=10)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=10, eta_min=1e-7)

    for epoch in range(10):  # loop over the dataset multiple times

        plot_loss = 0.0
        print_loss = 0.0
        for iter in range(train_size + 1):

            input_tensor, space_tensor, target_tensor = getInputAndTargerTensor(iter)

            loss = train(input_tensor, space_tensor, target_tensor, timeRnn_model,
                         spaceCnn_model, optimizer1, optimizer2, criterion)
            plot_loss += loss
            print_loss += loss

            if (iter + 1) % print_every == 0:  # print every 5000 mini-batches
                print('Epoch[%d,%5d, %d%%] -%s /step  -loss: %.4f ' % (
                    epoch + 1, iter + 1, (iter + 1) / (train_size + 1) * 100, timeSince(start, iter / train_size),
                    print_loss / print_every))
                print_loss = 0

            if iter % plot_every == 0:
                all_losses.append((plot_loss / plot_every))
                plot_loss = 0

        scheduler1.step()
        scheduler2.step()

    print('Training Finished')


# 验证
# 用测试集验证，求出混淆矩阵
def evaluate(time, space, c):
    total = 0

    list = np.zeros((c, c))

    with torch.no_grad():
        for i in range(70000, 100000):
            input_tensor, space_tensor, target_tensor = getInputAndTargerTensor(i)

            rnn_hidden = timeRnn_model.initHidden()
            input_length = input_tensor.size(0)

            rnn_output = torch.zeros((1, 64))
            for ei in range(input_length):
                rnn_output, encoder_hidden = timeRnn_model(
                    input_tensor[ei], rnn_hidden)

            space_output = spaceCnn_model(space_tensor, rnn_output)

            top_n, top_i = space_output.topk(1)
            predicted = top_i[0].item()

            total = total + 1
            list[target_tensor][predicted] += 1

    caculate(list, c, total)


def caculate(index, C, total):
    accuracy = 0
    recall = []
    precision = []
    f1_score = []
    for c in range(C):
        accuracy = accuracy + index[c][c]
        print(index[c][c])
        recall.append(100 * index[c][c] / (np.sum(index, axis=1)[c]))
        precision.append(100 * index[c][c] / (np.sum(index, axis=0)[c]))
        f1_score.append(2 * precision[c] * recall[c] / (precision[c] + recall[c]))

    print('测试集共计%d ,比例为0.3' % (train_size))
    print('Accuracy :', 100 * accuracy / total, accuracy)
    print('recall :', recall)
    print('precision :', precision)
    print('f1Score : ', f1_score)


# 验证
# 用测试集验证，求出混淆矩阵
def evaluate2(time, space, c):
    total = 0
    index = torch.zeros(c)

    tp, fp, tn, fn = 0, 0, 0, 0

    with torch.no_grad():
        for i in range(560000, 800000):
            input_tensor, space_tensor, target_tensor = getInputAndTargerTensor(i)

            rnn_hidden = timeRnn_model.initHidden()
            input_length = input_tensor.size(0)

            rnn_output = torch.zeros((1, 64))
            for ei in range(input_length):
                rnn_output, encoder_hidden = timeRnn_model(
                    input_tensor[ei], rnn_hidden)

            space_output = spaceCnn_model(space_tensor, rnn_output)

            top_n, top_i = space_output.topk(1)
            predicted = top_i[0].item()

            total = total + 1

            if target_tensor == 0 and predicted == 0:
                tn += 1
            if target_tensor == 0 and predicted != 0:
                fp += 1
            if target_tensor != 0 and predicted == 0:
                fn += 1
            if target_tensor != 0 and predicted != 0:
                tp += 1

    accuracy = 100 * (tp + tn) / total
    recall = 100 * (tp) / (tp + fn)
    precision = 100 * tp / (tp + fp)
    f1_score = 2 * precision * recall / (precision + recall)

    print('测试集共计%d ,比例为0.3' % (train_size))
    print('Accuracy :', accuracy)
    print('recall :', recall)
    print('precision :', precision)
    print('f1Score : ', f1_score)


# # 开始！
trainIter()

torch.save(timeRnn_model.state_dict(), 'r_model.pth')
torch.save(spaceCnn_model.state_dict(), 's_model.pth')

#
# # timeRnn_model.load_state_dict(torch.load('r_model.pth'))
# # spaceCnn_model.load_state_dict(torch.load('s_model.pth'))
plt.plot(all_losses)
torch.save(all_losses,'losses.pt')
plt.show()
evaluate(timeRnn_model, spaceCnn_model, 15)

# bot测试集共计30000 ,比例为0.3
# Accuracy : 99.9967 % (29999)
# recall : 100.0000  % (20184)
# precision : 99.9950  % (20184)
# f1Score : 99.9975  %


# ddosAccuracy : 99.9933 % (29998)
# recall : 99.9866 100.0000 100.0000 % (14969)
# precision : 100.0000 100.0000 99.9866 % (54)
# f1Score : 99.9933 100.0000 99.9933 % (14975)
