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


class Teacher:
    def __init__(self, data_size, validation_length, classes, device, gamma=0, output_size=1, hidden_size=256, channels=4):
        self.sample_size = 512
        self.classes = 10
        self.batch_size = 32
        self.memory = deque(maxlen=1000000)
        self.device = device

        self.q_network = QNetwork(data_size, validation_length, classes, hidden_size=hidden_size, channels=channels).to(device)
        self.optim = optim.Adam(self.q_network.parameters(), lr=1e-3)

    def get_batch(self, trainset, validset, preds):   
        valid_data = torch.cat([x[0] for x in validset], dim=0).to(self.device).unsqueeze(1)    # valid x 1 x w x h
        valid_labels = torch.zeros((len(validset), self.classes), dtype=torch.float, device=self.device)    # valid x 10
        for i, x in enumerate(validset):
            valid_labels[i][x[1]] = 1
            
        sample = random.sample(list(trainset), self.sample_size)
        train_data = torch.cat([x[0] for x in sample], dim=0).to(self.device).unsqueeze(1)    # sample_size x c x w x h
        train_labels = torch.tensor([x[1] for x in sample], dtype=torch.long, device=self.device)    # sample_size

        scores = self.q_network(valid_data, valid_labels, preds, train_data).squeeze(0)    # sample_size
        
        idx = torch.argmax(scores)    # 1
        
        batch = train_data[idx].unsqueeze(0), train_labels[idx].unsqueeze(0)    # B x c x w x h, B

        # print()
        # print('Teacher.get_batch')
        # print('valid_data (valid x c x w x h)', valid_data.shape)
        # print('valid_labels (valid x classes)', valid_labels.shape)
        # print('train_data (sample_size x c x w x h)', train_data.shape)
        # print('train_labels (sample_size x classes)', train_labels.shape)
        # print('scores (sample_size)', scores.shape)
        # print('batch data (1 x c x w x h)', batch[0].shape)
        # print('batch labels (1)', batch[1].shape)
        # print()

        return batch

    def get_random_batch(self, dataset):
        sample = random.sample(list(dataset), 1)
        data = torch.cat([x[0] for x in sample], dim=0).to(self.device).unsqueeze(1)
        labels = torch.tensor([x[1] for x in sample], dtype=torch.long, device=self.device)
        return data, labels

    def append_memory(self, preds, batch, next_preds, reward, done, valid_indices):
        self.memory.append((preds, batch, reward, valid_indices))

    def train(self, dataset):
        train_data = random.sample(self.memory, self.batch_size)

        preds = torch.tensor([x[0] for x in train_data], dtype=torch.float, device=self.device)     # train_size x valid_length x classes
        batch = torch.tensor([x[1] for x in train_data], dtype=torch.float, device=self.device)     # train_size x 1 x c x w x h
        reward = torch.tensor([x[2] for x in train_data], dtype=torch.float, device=self.device)    # train_size
        valid_indices = [x[3] for x in train_data]    # train_size x valid_length

        self.optim.zero_grad()

        v = []
        for i, vi in enumerate(valid_indices):
            validset = Subset(dataset, vi)
            valid_data = torch.cat([x[0] for x in validset], dim=0).to(self.device).unsqueeze(1)    # valid x 1 x w x h
            valid_labels = torch.zeros((len(validset), self.classes), dtype=torch.float, device=self.device)    # valid x 10
            for j, x in enumerate(validset):
                valid_labels[j][x[1]] = 1

            v.append(self.q_network(valid_data, valid_labels, preds[i], batch[i]))    # 1 x knowledge_size

        values = torch.cat(v, dim=0).squeeze(-1)    # train_size

        loss = (reward - values) ** 2

        loss.mean().backward()
        self.optim.step()

        # print()
        # print('Teacher.train')
        # print('preds (train_size x valid_length x classes)', preds.shape)
        # print('batch (train_size x 1 x c x w x h)', batch.shape)
        # print('reward (train_size)', reward.shape)
        # print('valid_data (valid x c x w x h)', valid_data.shape)
        # print('valid_labels (valid x classes)', valid_labels.shape)
        # print('values (train_size)', values.shape)
        # print('loss (train_size)', loss.shape)
        # print()


class QNetwork(nn.Module):
    def __init__(self, data_size, data_length, classes, hidden_size=512, channels=8, drop=.3):
        super(QNetwork, self).__init__()

        if len(data_size) != 3:
            raise Exception('Invalid data size', data_size)

        in_channels = data_size[0]
        out_kernal = int(data_size[-1] / 4)
        encoder_in = data_length * (channels*8 + 2*classes)
        action_in = hidden_size + channels * 8
        self.data_length = data_length

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),    # (4, 28, 28)
            nn.ReLU(),
            nn.Conv2d(channels, channels*2, 2, stride=2),    # (8, 14, 14)
            nn.ReLU(),
            nn.Conv2d(channels*2, channels*2, 3, padding=1),    # (8, 14, 14)
            nn.ReLU(),
            nn.Conv2d(channels*2, channels*4, 2, stride=2),    # (16, 7, 7)
            nn.ReLU(),
            nn.Conv2d(channels*4, channels*4, 3, padding=1),    # (16, 7, 7)
            nn.ReLU(),
            nn.Conv2d(channels*4, channels*8, out_kernal),    # (32, 1, 1)
            nn.ReLU()
        )
        
        self.encoder_net = nn.Sequential(
            nn.Linear(encoder_in, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.action_net = nn.Sequential(
            nn.Linear(action_in, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, data, labels, preds, action):
        if len(preds.shape) != 2:
            raise Exception('Invalid preds size', preds.shape)  # valid x classes
        if data.shape[0] != labels.shape[0] or data.shape[0] != preds.shape[0]:
            raise Exception('Invalid knowledge shapes', data.shape[0], labels.shape[0], preds.shape[0])  # valid
        if len(action.shape) != 4:
            raise Exception('Invalid action size', action.shape)    # B x c x w x h

        batch = action.shape[0]
        valid_length = labels.shape[0]

        data_rep = self.conv(data).view(self.data_length, -1)    # valid x c

        encoder_in = torch.cat((data_rep, labels, preds), dim=-1).view(1, -1)  # 1 x valid * (c + 2*classes)

        encoder_out = self.encoder_net(encoder_in).squeeze(0)  # hidden
        encoder_stack = encoder_out.repeat(batch, 1)  # B x hidden

        action_rep = self.conv(action).view(batch, -1)  # B x c

        action_in = torch.cat((encoder_stack, action_rep), dim=-1)  # B x hidden + c

        out = self.action_net(action_in)    # B

        # print()
        # print('QNetwork.forward')
        # print('data (valid x c x w x h)', data.shape)
        # print('labels (valid x 10)', labels.shape)
        # print('preds (valid x 10)', preds.shape)
        # print('action (B x c x w x h)', action.shape)
        # print('data_rep (valid x C)', data_rep.shape)
        # print('encoder_in (1 x valid * (c + 2*classes))', encoder_in.shape)
        # print('encoder_out (1 x hidden)', encoder_out.shape)
        # print('encoder_stack (B x hidden)', encoder_stack.shape)
        # print('action_rep (B x C)', action_rep.shape)
        # print('action_in (B x hidden + C)', action_in.shape)
        # print('out (B x 1)', out.shape)
        # print()

        return out
