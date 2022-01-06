import torch.nn as nn
import torch
import torchvision
import time
import math


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class RNN(nn.Module):
    def __init__(self, hidden_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.relu = nn.ReLU()

        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 128)

    def forward(self, input, hidden):
        output = self.relu(input)
        output=output.view(1,1,-1)
        output, hidden = self.gru(output, hidden)
        output=self.relu(self.out(output))

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class Space(nn.Module):
    def __init__(self):
        super(Space, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x, rnn_output):
        # 26*26 13
        x = self.pool(self.relu(self.conv1(x)))
        # 11*11 5
        x = self.pool(self.relu(self.conv2(x)))
        # 4*4
        x = self.relu(self.conv3(x))
        # 64*4*4=1024
        x = torch.flatten(x, 1)  # 除去批次维度，其他展平 [N,1024]
        # 1*64

        x = self.relu(self.fc1(x))

        rnn_output=rnn_output.view(1,-1)
        # 128+128=256
        x = torch.cat((x, rnn_output), 1)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x






