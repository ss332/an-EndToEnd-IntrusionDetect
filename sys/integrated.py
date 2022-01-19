import numpy as np
import model_v1
import torch
import torch.nn as nn

from captum.attr import IntegratedGradients

timeRnn_model = model_v1.RNN(320)
spaceCnn_model = model_v1.Space()




def getInputAndTargerTensor(i):

    flow_name = r'D:\sessions\{}\session{}.pt'.format('files1', 900)

    # 归一化
    # n*320
    input_tensor = torch.load(flow_name)

    # 1*1024
    space_tensor = torch.zeros(1024)
    size = 0

    for k in range(input_tensor.size(0)):
        x = input_tensor[k]
        for j in range(x.size(0)):
            if x[j] != 0 and size < 1024:
                space_tensor[size] = x[j]
                size = size + 1

    # (b,c,w,h)
    space_tensor = space_tensor.view(1, 1, 32, 32)
    target_tensor = torch.tensor([1,0])

    return input_tensor, space_tensor, target_tensor


ig = IntegratedGradients(spaceCnn_model)
input_tensor, space_tensor,target_tensor=getInputAndTargerTensor(900)

rnn_hidden = timeRnn_model.initHidden()
rnn_output=timeRnn_model(input_tensor[0],rnn_hidden)

baselin1=torch.zeros((1,1,32,32))
baselin2=torch.zeros((1,320))

input=(space_tensor,rnn_output)
baseline= (baselin1,baselin2)

attributions, delta = ig.attribute(*input, *baseline, target=2)

print('IG Attributions:', attributions)
print('Convergence Delta:', delta)