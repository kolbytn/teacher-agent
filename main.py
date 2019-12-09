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
import sys
import getopt

from teacher import Teacher
from student import Student


def train_teacher(teacher, student, dataset, valid_size, train_length=1000, episode_length=2000):
    train_freq = 5
    train_start = 100
    epsilon = 1
    epsilon_decay = .9999
    steps = 0
    save_freq = 3

    for epoch in range(train_length):
        # Do new training/val split at each epoch
        train_indices = list(range(len(dataset)))
        valid_indices = random.sample(train_indices, valid_size)
        train_indices = [x for x in train_indices if x not in valid_indices]

        trainset = Subset(dataset, train_indices)
        validset = Subset(dataset, valid_indices)
        student.reset()

        cum_reward = []
        preds, last_accuracy = student.eval(validset)
        for timestep in range(episode_length):

            if random.random() < epsilon:
                batch_data, batch_labels = teacher.get_random_batch(trainset)
            else:
                batch_data, batch_labels = teacher.get_batch(trainset, validset, preds)

            epsilon *= epsilon_decay

            student.train((batch_data, batch_labels))

            next_preds, accuracy = student.eval(validset)
            reward = accuracy
            cum_reward.append(reward)

            teacher.append_memory(preds.cpu().detach().numpy(), 
                                  batch_data.cpu().detach().numpy(), 
                                  next_preds.cpu().detach().numpy(), 
                                  reward, 
                                  timestep == episode_length - 1, 
                                  valid_indices)

            preds = next_preds
            last_accuracy = accuracy
            steps += 1

            if steps > train_start and steps % train_freq == 0:
                teacher.train(dataset)

        print("Epoch: {}, Timestep: {}, Accuracy: {}, Reward: {}".format(epoch, timestep, accuracy, sum(cum_reward) / len(cum_reward)))

        if epoch % save_freq == 0:
            torch.save(teacher.q_network.state_dict(), '11_25_19_weights/q_network' + str(epoch))


if __name__ == '__main__':
    device = 'cuda'
    valid_size = 512
    episode_length = 2000
    train_length = 1000000 
    gamma = 0
    batch_size = 1

    try:
        opts, args = getopt.getopt(sys.argv[1:], "e:t:g:b:")
    except getopt.GetoptError:
        print("Invalid args")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-e':
            episode_length = int(arg)
            print(episode_length)
        if opt == '-t':
            train_length = int(arg)
            print(train_length)
        if opt == '-g':
            gamma = float(arg)
            print(gamma)
        if opt == '-b':
            batch_size = int(arg)
            print(batch_size)

    dataset = torchvision.datasets.MNIST('/tmp', download=True, transform=transforms.ToTensor())
    teacher = Teacher(dataset[0][0].shape, valid_size, 10, device, gamma=gamma, output_size=batch_size)
    student = Student(dataset[0][0].shape, device)

    train_teacher(teacher, student, dataset, valid_size, episode_length=episode_length, train_length=train_length)
