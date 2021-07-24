from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import torch
import unicodedata
import string
import torch.nn as nn
import random
import math
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def findFiles(path): return glob.glob(path)


print(findFiles('data/names/*.txt'))
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


# unicode 转ASCII
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn'
                   and c in all_letters)


print(unicodeToAscii('Ślusàrski'))

# 创建category_line字典，对应每种语言一列名字
category_lines = {}
all_categories = []


# 读取文件，每一行的名字写入lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)
print(category_lines['Italian'][:7])


# 将lines中的名字转化为tensor张量
# 返回字母在全letters中的索引
def letterToIndex(letter):
    return all_letters.find(letter)


# 将字母转换成<1*n_letters>张量
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


# 将lines中的名字line转化为<line_length*1*n_letters> 中间的1是批次，batch为1
def lineToTensor(line):
    line_length = len(line)
    tensor = torch.zeros(line_length, 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


print(letterToTensor('J'))
print(lineToTensor('Jones').size())


# 创建循环神经网络

class LSTMRnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMRnn, self).__init__()

        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden):
        output, hidden = self.lstm1(input, hidden)
        output = self.h2o(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, n_hidden), torch.zeros(1, 1, n_hidden)


n_hidden = 128

lstmRnn = LSTMRnn(n_letters, n_hidden, n_categories)
line_tensor = lineToTensor('Albert')

hidden = (torch.zeros(1, 1, n_hidden), torch.zeros(1, 1, n_hidden))
output, next_hidden1 = lstmRnn(line_tensor[0].unsqueeze(0), hidden)
print(output)


def categoryFromOutput(output):
    top_v, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category=', category, '/line=', line)

# 训练神经网络
criterion = nn.NLLLoss()
learning_rate = 0.001
# optimizer = optim.SGD(lstmRnn.parameters(), lr=learning_rate, momentum=0.9)
optimizer = torch.optim.RMSprop(lstmRnn.parameters(), lr=learning_rate)


def train(category_tensor, line_tensor):
    global output
    hidden = lstmRnn.initHidden()
    optimizer.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden1 = lstmRnn(line_tensor[i].unsqueeze(0), hidden)

    loss = criterion(output.squeeze(0), category_tensor)
    loss.backward()
    optimizer.step()

    return output, loss.item()


n_iters = 100000
print_every = 5000
plot_every = 1000

current_loss = 0
all_losses = []


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = 'yes' if guess == category else 'x(%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100,
                                                timeSince(start), loss, line, guess, correct))

    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

plt.figure()
plt.plot(all_losses)
plt.show()
# 评价模型性能
# 使用混淆矩阵存储正确的猜测，行代表实际的类别，列代表模型预测，猜的越准对角线应该越亮
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 100000


def evaluate(line_tensor):
    global output
    hidden = lstmRnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = lstmRnn(line_tensor[i].unsqueeze(0), hidden)

    return output


for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
plt.show()


def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])


predict('Dovesky')
predict('Jackson')
predict('Satoshi')