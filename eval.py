import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
import numpy as np
from scipy import stats
import random
import csv
import os
import matplotlib.pyplot as plt

from teacher import Teacher
from student import Student


def eval_teacher(teacher, student, dataset, rand, valid_size, acc_reward, episode_length, 
                 res_path, epochs=1000):
    rewards = [[] for _ in range(episode_length)]

    for epoch in range(epochs):

        # Do new training/val split at each epoch
        train_indices = list(range(len(dataset)))
        valid_indices = random.sample(train_indices, valid_size)
        train_indices = [x for x in train_indices if x not in valid_indices]
        trainset = Subset(dataset, train_indices)
        validset = Subset(dataset, valid_indices)

        student.reset()
        episode_reward = []
        preds, last_accuracy = student.eval(validset)
        for timestep in range(episode_length):

            if rand:
                batch_data, batch_labels = teacher.get_random_batch(trainset)
            else:
                batch_data, batch_labels = teacher.get_batch(trainset, validset, preds)

            student.train((batch_data, batch_labels))

            next_preds, accuracy = student.eval(validset)

            if acc_reward:
                reward = accuracy
            else:
                reward = accuracy - last_accuracy

            episode_reward.append(reward.item())

            preds = next_preds
            last_accuracy = accuracy

            print("Epoch: {}, Timestep: {}, Accuracy: {}, Reward: {}".format(epoch, timestep, accuracy, sum(episode_reward) / len(episode_reward)))

        save_results(episode_reward, res_path, episode_length)

    return rewards


def test_mse(teachers, students, dataset, res_path, epochs=100, device='cuda'):

    for epoch in range(epochs):
        # Do new training/val split at each epoch
        train_indices = list(range(len(dataset)))
        valid_indices = random.sample(train_indices, valid_size)
        train_indices = [x for x in train_indices if x not in valid_indices]
        trainset = Subset(dataset, train_indices)
        validset = Subset(dataset, valid_indices)

        batch_data, batch_labels = teachers[0].get_random_batch(trainset)
        valid_data = torch.cat([x[0] for x in validset], dim=0).to(device).unsqueeze(1)    # valid x 1 x w x h
        valid_labels = torch.zeros((len(validset), 10), dtype=torch.float, device=device)    # valid x 10
        for i, x in enumerate(validset):
            valid_labels[i][x[1]] = 1

        mse_list = []
        for teacher, student in zip(teachers, students):
            student.reset()
            preds, last_accuracy = student.eval(validset)
  
            scores = teacher.q_network(valid_data, valid_labels, preds, batch_data).squeeze(0).item()    # sample_size

            student.train((batch_data, batch_labels))

            next_preds, accuracy = student.eval(validset)

            reward = (accuracy).item()

            mse = (scores - reward) ** 2

            mse_list.append(mse)

            print("Epoch: {}, MSE: {}".format(epoch, mse))
            
        save_results(mse_list, res_path, len(teachers))


def save_results(rewards, name, episode_length):

    old_results = [[] for _ in range(episode_length)]
    if os.path.exists(name):
        with open(name) as file:
            reader = csv.reader(file, delimiter=',')
            for i, t in enumerate(reader):
                for r in t:
                    old_results[i].append(r)

    with open(name, "w", 1) as file:
        csv_writer = csv.writer(file, delimiter=",")
        for i, t in enumerate(rewards):
            old_results[i].append(t)
            csv_writer.writerow(old_results[i])


def plot_acc_results(paths, name):
    
    plt.clf()
    for c, l, p in zip(['green', 'orange'], ['Random Teacher', 'RL Teacher'], paths):
        rewards = []
        with open(p) as file:
            reader = csv.reader(file, delimiter=',')
            for t in reader:
                rewards.append([])
                for r in t:
                    rewards[-1].append(float(r))

        rewards = rewards[:200]
        avg = [sum(x) / len(x) for x in rewards]
        high = [sum(x) / len(x) + stats.tstd(x) / 4 for x in rewards]
        low = [sum(x) / len(x) - stats.tstd(x) / 4 for x in rewards]

        plt.plot(avg, color=c, label=l)
        plt.fill_between(list(range(len(low))), low, high, alpha=.2, color=c)

    plt.xlabel('Student Steps')
    plt.ylabel('Validation Accuracy')
    plt.title('Student Performance Comparison')
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def plot_mse_results(path, name):
    
    plt.clf()
    loss = []
    with open(path) as file:
        reader = csv.reader(file, delimiter=',')
        for t in reader:
            loss.append([])
            for r in t:
                loss[-1].append(float(r))

    avg = [sum(x) / len(x) for x in loss]
    high = [sum(x) / len(x) + stats.tstd(x) / 4 for x in loss]
    low = [sum(x) / len(x) - stats.tstd(x) / 4 for x in loss]

    plt.plot(list(range(0, 12001, 1200)), avg, color='orange')
    plt.fill_between(list(range(0, 12001, 1200)), low, high, alpha=.2, color='orange')

    plt.xlabel('Teacher Steps')
    plt.ylabel('MSE Loss')
    plt.title('Teacher Loss')
    plt.savefig(name, bbox_inches='tight')


if __name__ == '__main__':
    device = 'cuda'
    valid_size = 512
    episode_length = 2000
    gamma = 0
    batch_size = 1

    dataset = torchvision.datasets.MNIST('/tmp', download=True, transform=transforms.ToTensor())

    # MSE over time
    paths = [ 'weights/' ]  # TODO paths to weights
    res_path = 'acc_mse2.csv'
    hidden = 512
    channels = 8

    teachers = []
    students = []
    for p in paths:
        teacher = Teacher(dataset[0][0].shape, valid_size, 10, device, gamma=gamma, output_size=batch_size, hidden_size=hidden, channels=channels)
        student = Student(dataset[0][0].shape, device)
        
        teacher.q_network.load_state_dict(torch.load(p))
        
        teachers.append(teacher)
        students.append(student)

    test_mse(teachers, students, dataset, res_path, epochs=90)
    plot_mse_results(res_path, 'mse2.png')

    #-------------------------------------------------------------------------------------------

    Accuracy Reward
    path = 'weights/'  # TODO path to weights
    hidden = 512
    channels = 8
    acc_reward = True
    acc_base_path = 'acc_base.csv'
    acc_res_path = 'acc_res.csv'
    teacher = Teacher(dataset[0][0].shape, valid_size, 10, device, gamma=gamma, output_size=batch_size, hidden_size=hidden, channels=channels)
    teacher.q_network.load_state_dict(torch.load(path))
    student = Student(dataset[0][0].shape, device)

    baseline = eval_teacher(teacher, student, dataset, True, valid_size, acc_reward, episode_length, acc_base_path)
    result = eval_teacher(teacher, student, dataset, False, valid_size, acc_reward, episode_length, acc_res_path)

    plot_acc_results([acc_base_path, acc_res_path], 'acc.png')
