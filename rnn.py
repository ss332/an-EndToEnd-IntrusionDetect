from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import torch
import unicodedata
import string
import torch.nn as nn
import random
import torch.optim as optim
import math
import time


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


# 将lines中的名字line转化为<line_length*1*n_letters>
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
        self.lstm2 = nn.LSTM(hidden_size, output_size, num_layers=1)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden1, hidden2):
        output1, hidden1 = self.lstm1(input, hidden1)

        output, hidden2 = self.lstm2(output1, hidden2)
        output = self.softmax(output)
        return output, hidden1, hidden2

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


n_hidden = 128

lstmRnn = LSTMRnn(n_letters, n_hidden, n_categories)
line_tensor = lineToTensor('Albert')
hidden_1 = (torch.zeros(1, 1, n_hidden), torch.zeros(1, 1, n_hidden))
hidden_2 = torch.zeros(1, 1, n_categories), torch.zeros(1, 1, n_categories)
output, next_hidden1, next_hidden2 = lstmRnn(line_tensor[0].unsqueeze(0), hidden_1, hidden_2)
print(output)


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line).unsqueeze(0)
    return category, line, category_tensor, line_tensor


for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category=', category, '/line=', line)

# 训练神经网络
criterion = nn.NLLLoss()
learning_rate = 0.005
optimizer = optim.SGD(lstmRnn.parameters(), lr=learning_rate, momentum=0.9)
optimizer2 = torch.optim.RMSprop(lstmRnn.parameters(), lr=learning_rate)


def train(category_tensor, line_tensor):


    global output
    hidden1 = (torch.zeros(1, 1, n_hidden), torch.zeros(1, 1, n_hidden))
    hidden2 = torch.zeros(1, 1, n_categories), torch.zeros(1, 1, n_categories)

    optimizer.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden1, hidden2 = lstmRnn(line_tensor[i].unsqueeze(0), hidden1, hidden2)

    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()

    return output, loss.item()
