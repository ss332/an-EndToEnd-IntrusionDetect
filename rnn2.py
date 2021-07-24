from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # Plus EOS marker
import time
import math


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def findFiles(path): return glob.glob(path)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def readLines(filename):
    with open(filename, encoding='utf-8') as some_file:
        return [unicodeToAscii(line.strip()) for line in some_file]


category_lines = {}
all_categories = []
for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

print('# categories:', n_categories, all_categories)
print(unicodeToAscii("O'Néàl"))


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(n_categories + input_size + hidden_size, hidden_size)
        self.o2o = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.logSoftmax = nn.LogSoftmax(dim=2)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 2)
        output, hidden = self.gru(input_combined)
        output = self.o2o(output)
        output = self.dropout(output)
        output = self.logSoftmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1,1, self.hidden_size)


# 从列表中随机选择一项
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


# 随机选择一种语言，拿去该语言下的随机一个名字
def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line


def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor


def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor


def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)


def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor


criterion = nn.NLLLoss()
learning_rate = 0.0005
rnn = RNN(n_letters, 128, n_letters)
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)


def train(category_tensor, input_line_tensor, target_line_tensor):
    global output
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

    optimizer.zero_grad()
    loss = 0
    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor.unsqueeze(0), input_line_tensor[i].unsqueeze(0), hidden)

        l = criterion(output[0], target_line_tensor[i])
        loss += l

    loss.backward()
    optimizer.step()

    return output, loss.item() / input_line_tensor.size(0)


n_iters = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0

start = time.time()

for iter in range(1, n_iters + 1):
    output, loss = train(*randomTrainingExample())
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append((total_loss / plot_every))

        total_loss = 0

plt.figure()
plt.plot(all_losses)
max_length = 20


def sample(category, start_letter='A'):
    with torch.no_grad():
        category_tensor = categoryTensor(category)
        input = inputTensor(start_letter)
        hidden = rnn.initHidden()
        output_name = start_letter
        for i in range(max_length):
            output, hidden = rnn(category_tensor.unsqueeze(0), input[0].unsqueeze(0), hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)
        return output_name


def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))

samples('Russian', 'RUS')

samples('German', 'GER')

samples('Spanish', 'SPA')

samples('Chinese', 'CHIW')
# 运行结果
# O'Neal
# 0m 22s (5000 5%) 2.3482
# 0m 44s (10000 10%) 2.2950
# 1m 6s (15000 15%) 2.3451
# 1m 28s (20000 20%) 1.8850
# 1m 50s (25000 25%) 3.7276
# 2m 12s (30000 30%) 1.9977
# 2m 34s (35000 35%) 1.2923
# 2m 56s (40000 40%) 0.5017
# 3m 19s (45000 45%) 1.9656
# 3m 43s (50000 50%) 2.2989
# 4m 6s (55000 55%) 1.3933
# 4m 29s (60000 60%) 1.7699
# 4m 53s (65000 65%) 1.6580
# 5m 16s (70000 70%) 1.2088
# 5m 40s (75000 75%) 2.0686
# 6m 4s (80000 80%) 0.8134
# 6m 28s (85000 85%) 2.3321
# 6m 51s (90000 90%) 0.5563
# 7m 13s (95000 95%) 0.8990
# 7m 36s (100000 100%) 2.4100
# Rithev
# Ubelyan
# Shelin
# Groen
# Ester
# Rosen
# Sala
# Parez
# Alamarano
# Cao
# Hua
# Iwa
# Wan