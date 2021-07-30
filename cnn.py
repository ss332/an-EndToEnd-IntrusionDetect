import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
import time
import math

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # (3,32,32)->(6,28,28)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # 除去批次维度，其他展平 [N,400]
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


lenet = LeNet()
train_size = len(trainset)
size = int(train_size / 4)
print(train_size, size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lenet.parameters(), lr=0.001)
start = time.time()


def train():
    for epoch in range(3):
        running_loss = 0.0
        for i in range(size):
            if i == 12500:
                break
            images = torch.zeros(size=(4, 3, 32, 32))
            labels = torch.zeros(4, dtype=torch.long)

            for j in range(4):
                images[j] = trainset[i * 4 + j][0]
                labels[j] = trainset[i * 4 + j][1]

            optimizer.zero_grad()
            outputs = lenet(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % 2000 == 0:
                print('[%d,%5d] -%s /step  -loss: %.3f' % (epoch + 1, i + 1, timeSince(start), running_loss / 2000))
                running_loss = 0


# train()
print('Finishing training')
PATH = './cifar_net.pth'
# torch.save(lenet.state_dict(), PATH)
net = LeNet()
net.load_state_dict(torch.load(PATH))


def evaluate():
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(len(testset)):
            image = testset[i][0].unsqueeze(0)
            label = testset[i][1]
            outputs = net(image)
            _, predicted = torch.max(outputs.data, 1)
            total = total + 1

            if label == predicted.item():
                correct += 1

    print('Accuracy of the network on the %d test images: %d %% (%d)' % (len(testset),
                                                                         100 * correct / total, correct))


evaluate()


def classPerform():
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for i in range(len(testset)):
            image = testset[i][0].unsqueeze(0)
            label = testset[i][1]
            outputs = net(image)
            _, predicted = torch.max(outputs.data, 1)
            if label == predicted.item():
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))


classPerform()
# 50000 12500
# Finishing training
# Accuracy of the network on the 10000 test images: 60 % (6020)
# Accuracy for class plane is: 64.8 %
# Accuracy for class car   is: 67.0 %
# Accuracy for class bird  is: 37.2 %
# Accuracy for class cat   is: 43.0 %
# Accuracy for class deer  is: 55.5 %
# Accuracy for class dog   is: 43.2 %
# Accuracy for class frog  is: 67.0 %
# Accuracy for class horse is: 72.7 %
# Accuracy for class ship  is: 76.2 %
# Accuracy for class truck is: 75.4 %
