import math
import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

'''
Author: Rahul Sajnani
Date  : 1 March 2021
'''
def conv_block(in_channels, out_channels, kernel, padding):

    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=padding, bias = False),
            nn.InstanceNorm2d(out_channels),
            nn.ELU(inplace=False),
        )



class convrelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, num_layers = 2):
        #super().__init__()
        super(convrelu, self).__init__()
        self.conv_layers = self.create_block(in_channels, out_channels, kernel, num_layers)

        # Skip connection
        if num_layers == 1:
            self.skip = False
        else:
            self.identity_block = self.create_block(in_channels, out_channels, 1, 1)
            self.skip = True


    def create_block(self, in_channels, out_channels, kernel, num_layers):

        layers = []
        mid = num_layers // 2
        last_out_layer = in_channels
        padding = kernel // 2
        for i in range(num_layers):

            if i != mid:
                layers.append(conv_block(last_out_layer, last_out_layer, kernel, padding))
            else:
                layers.append(conv_block(last_out_layer, out_channels, kernel, padding))
                last_out_layer = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv_layers(x)

        if self.skip:
            out = out + self.identity_block(x)

        return out


class DQN(nn.Module):

    def __init__(self, in_channels=3, num_actions=6):
        super(DQN, self).__init__()

        self.conv1 = convrelu(in_channels, 64, 3, 2)
        self.conv2 = convrelu(64, 128, 3, 2)
        self.conv3 = convrelu(128, 64, 3, 1)

        self.pool = nn.MaxPool2d(3)

        self.fc4 = nn.Linear(2881, 512)
        self.fc5 = nn.Linear(512, 64)
        self.fc6 = nn.Linear(64, num_actions)

    def forward(self, x, x_sensor):
        '''
        Inputs:
            x       - B, C, H, W   - Batch of images
            x_senor - B, 1         - Signal strength
        '''

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = torch.hstack((x, x_sensor))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))

        return self.fc6(x)


if __name__=="__main__":

    x = torch.randn(2, 3, 256, 144)
    x_sensor = torch.randn(2, 1)
    net = DQN(3, 6)
    out = net(x, x_sensor)

    print(out.shape)
