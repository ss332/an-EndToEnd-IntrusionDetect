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
# 训练集要打乱，采用小卷积核3*3，深度提高，遍历10次数据集就差不多了
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 8

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(images.size())
print(images.dtype)
print(labels)

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


class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # 30*30 15
        x = self.pool(self.relu(self.conv1(x)))
        # 13*13 6
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = torch.flatten(x, 1)  # 除去批次维度，其他展平 [N,1024]
        x = self.relu(self.fc1(x))
        x = self.fc3(x)
        return x


train_size = len(trainset)
size = int(train_size / 4)
print(train_size, size)

# 随机权重平均，使用多个不同更新获得的权重的平均值作为预测模型权重，泛化性能更好
net = Net2()
swa_model = optim.swa_utils.AveragedModel(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
swa_scheduler = optim.swa_utils.SWALR(optimizer, swa_lr=0.0005)

start = time.time()


def train1():
    for epoch in range(15):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (i+1) % 2000 == 0:  # print every 2000 mini-batches
                print('[%d,%5d] -%s /step  -loss: %.3f ' % (epoch + 1, i + 1, timeSince(start), running_loss / 2000))
                running_loss = 0.0
        if epoch > 6:
            swa_model.update_parameters(net)
            swa_scheduler.step()
        else:
            scheduler.step()

    print('Finished Training')
    # 没有batch  batch normalization，可以不用
    optim.swa_utils.update_bn(trainloader, swa_model)


def train():
    for epoch in range(10):
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
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % 2000 == 0:
                print('[%d,%5d] -%s /step  -loss: %.3f' % (epoch + 1, i + 1, timeSince(start), running_loss / 2000))
                running_loss = 0



train1()
print('Finishing training')
# PATH = './cifar_net.pth'
# torch.save(net.state_dict(), PATH)
# net = Net2()
# net.load_state_dict(torch.load(PATH))


def evaluate():
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(len(testset)):
            image = testset[i][0].unsqueeze(0)
            label = testset[i][1]
            outputs = swa_model(image)
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
            outputs = swa_model(image)
            _, predicted = torch.max(outputs.data, 1)
            if label == predicted.item():
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))


classPerform()

# 50000 12500
# Finishing training -lenet
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

# SWA+NET2+SCHEDULE BATCH=8
# Accuracy of the network on the 10000 test images: 70 % (7029)
# Accuracy for class plane is: 75.3 %
# Accuracy for class car   is: 81.5 %
# Accuracy for class bird  is: 55.0 %
# Accuracy for class cat   is: 51.0 %
# Accuracy for class deer  is: 67.3 %
# Accuracy for class dog   is: 62.7 %
# Accuracy for class frog  is: 75.9 %
# Accuracy for class horse is: 77.3 %
# Accuracy for class ship  is: 78.8 %
# Accuracy for class truck is: 78.1 %
