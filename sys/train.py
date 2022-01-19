import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import time
import torch.optim as optim
import model_v2
import numpy as np
from utils import timeSince
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import data
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'


set_size = 800000
train_size = 560000
test_size = 240000
print_every = 200
plot_every = 500
# 记录损失曲线
all_losses = []
start = time.time()

hidden_size = 320


# 训练，循环训练集10个纪元
def trainIter(model, loader):
    print('Training start!')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # 学习率衰减计划，按cos曲线衰减，最大周期为10

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7)

    for epoch in range(10):  # loop over the dataset multiple times

        plot_loss = 0.0
        print_loss = 0.0
        iter=0
        for session, packets, p_len, label_idx in loader:

            pred = model(session, packets, p_len)

            optimizer.zero_grad()

            loss = 0

            loss += criterion(pred, label_idx)

            loss.backward()
            optimizer.step()

            loss = loss.item()
            plot_loss += loss
            print_loss += loss

            if (iter + 1) % print_every == 0:  # print every 5000 mini-batches
                print('Epoch[%d,%5d, %d%%] -%s /step  -loss: %.4f ' % (
                    epoch + 1, iter + 1, (iter + 1) / (train_size/32) * 100, timeSince(start, iter / train_size),
                    print_loss / print_every))
                print_loss = 0

            if iter % plot_every == 0:
                all_losses.append((plot_loss / plot_every))
                plot_loss = 0

            iter+=1

        scheduler.step()

    print('Training Finished')


# 验证
# 用测试集验证，求出混淆矩阵
def evaluate(model, c,loader):
    total = 0

    list = np.zeros((c, c))

    with torch.no_grad():
        for session, packets, p_len, label_idx in loader:

            pred = model(session, packets, p_len)

            top_n, top_i = pred.topk(1)
            # predicted = top_i[0].item()

            total = total + 32
            for p,l in zip(top_i,label_idx):

                list[l.item()][p] += 1

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


def main():
    print(f'Using {device} device')
    net = nn.DataParallel(model_v2.ArcNet()).to(device)
    # # 开始！
    path = r'D:\sessions'
    training_data = data.IDSDataset('all_train.csv', path, )
    test_data = data.IDSDataset('all_test.csv', path)

    train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True,num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True,num_workers=4)

    trainIter(net, train_dataloader)
    evaluate(net,15,test_dataloader)

    torch.save(net.state_dict(), 'model.pth')

    # model.load_state_dict(torch.load('model.pth'))

    plt.plot(all_losses)
    torch.save(all_losses, 'losses.pt')
    plt.show()



if __name__ == '__main__':
    main()
