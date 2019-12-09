import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torchvision
import torch.optim as optim
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
import numpy as np
import random
import gc
from collections import deque
from itertools import chain


class Student:
    def __init__(self, data_size, device):
        #Store dataset so we have access
        self.data_size = data_size
        self.device = device
        #Create ConvNetwork
        self.network = ConvNetwork(data_size).to(device)
        #Use Cross Entropy Loss function
        self.objective = nn.CrossEntropyLoss()
        #Use Adam optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3)
        
    def reset(self):
        #Reinstantiate new ConvNetwork
        self.network = ConvNetwork(self.data_size).to(self.device)
        #Zero Gradients of optimizer
        self.optimizer.zero_grad()

    def train(self, batch):
        x, y_truth = batch[0].to(self.device), batch[1].to(self.device)
        self.network.train()
        #Zero optimizer gradient
        self.optimizer.zero_grad()
        #Get network predictions
        y_hat = self.network(x)
        #Compute Loss, call .backward()
        loss = self.objective(y_hat, y_truth)
        loss.backward()
        #optimizer step
        self.optimizer.step()

    def compute_accuracy(self, y_hat, y_truth):
        return (y_hat.argmax(1) == y_truth).float().mean()
    
    def eval(self, validation_set):
        #dim x should len x channels x h x w
        x = torch.cat([x[0] for x in validation_set], dim=0).to(self.device).unsqueeze(1)
        #y_truth should be len
        y_truth = torch.tensor([x[1] for x in validation_set], dtype=torch.long, device=self.device)

        self.network.eval()
        y_hat = self.network(x)
        accuracy = self.compute_accuracy(y_hat, y_truth)

        return y_hat, accuracy


class ConvNetwork(nn.Module):
    def __init__(self, data_size):
        super(ConvNetwork, self).__init__()
        c, h, w = data_size
        self.linear_in = h * w * 10
        output = 10

        self.conv = nn.Sequential(
                nn.Conv2d(c, 10, (3,3), padding=(1,1)),
                nn.BatchNorm2d(10),
                nn.ReLU(),
                nn.Conv2d(10, 10, (3,3), padding=(1,1)),
                nn.BatchNorm2d(10),
                nn.ReLU(),
                nn.Conv2d(10, 10, (3,3), padding=(1,1)),
                nn.BatchNorm2d(10),
                nn.ReLU(),
                nn.Conv2d(10, 10, (3,3), padding=(1,1)),
                nn.BatchNorm2d(10),
                nn.ReLU(),
                nn.Dropout(.2)
        )

        self.linear = nn.Linear(self.linear_in, output)

    def forward(self, x):
        conv_out = self.conv(x).view(-1, self.linear_in)
        return self.linear(conv_out)
        