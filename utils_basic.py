import numpy as np
import random
import torch
from sklearn.metrics import roc_auc_score
import torchvision
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import pandas as pd
import PIL
import logging
import torchattacks
import sys, os
sys.path.append('.')

def load_dataset_setting(task):
    if task == 'mnist':
        BATCH_SIZE = 100
        N_EPOCH = 100
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        trainset = torchvision.datasets.MNIST(root='./raw_data/', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./raw_data/', train=False, download=False, transform=transform)
        is_binary = False
        need_pad = False
        from model_lib.mnist_cnn_model import Model, troj_gen_func, random_troj_setting
#############################################################################################        
    elif task == 'cifar10':
        BATCH_SIZE = 100
        N_EPOCH = 100
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./raw_data/', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./raw_data/', train=False, download=False, transform=transform)
        is_binary = False
        need_pad = False
        from model_lib.cifar10_cnn_model import Model , troj_gen_func, random_troj_setting
        from model_lib.resnet import ResNet18
        from model_lib.preact_resnet import PreActResNet18
#############################################################################################
    elif task == 'gtsrb':
        BATCH_SIZE = 100
        N_EPOCH = 100
        transform = transforms.Compose([
            transforms.Resize([112, 112]),
            transforms.ToTensor()
            ])
    
        trainset = torchvision.datasets.GTSRB(root='./raw_data/', split='train', download=True, transform=transform)
        testset = torchvision.datasets.GTSRB(root='./raw_data/', split='test', download=True, transform=transform)
        print(f'Number of training samples: {len(trainset)}')
        print(f'Number of test samples: {len(testset)}')
        is_binary = False
        need_pad = False
        from model_lib.gtsrb_cnn_model import  troj_gen_func, random_troj_setting
        from model_lib.gtsrb_cnn_model import Model
#############################################################################################
    elif task == 'chest':
        BATCH_SIZE = 32
        N_EPOCH = 100
        from model_lib.chest_cnn_model import Model, troj_gen_func, random_troj_setting, data_loader
        dataset = data_loader(root_dir='raw_data\chest')

        trainset = dataset['train']
        print(len(trainset))

        testset = dataset['test']
        is_binary = False
        need_pad = False
#############################################################################################

    else:
        raise NotImplementedError("Unknown task %s"%task)

    return BATCH_SIZE, N_EPOCH, trainset, testset, is_binary, need_pad, Model, troj_gen_func, random_troj_setting


class BackdoorDataset(torch.utils.data.Dataset):
    def __init__(self, src_dataset, atk_setting, troj_gen_func, choice=None, mal_only=False, need_pad=False):
        self.src_dataset = src_dataset
        self.atk_setting = atk_setting
        self.troj_gen_func = troj_gen_func
        self.need_pad = need_pad

        self.mal_only = mal_only
        if choice is None:
            choice = np.arange(len(src_dataset))
        self.choice = choice
        inject_p = atk_setting[5]
        self.mal_choice = np.random.choice(choice, int(len(choice)*inject_p), replace=False)

    def __len__(self,):
        if self.mal_only:
            return len(self.mal_choice)
        else:
            return len(self.choice) + len(self.mal_choice)

    def __getitem__(self, idx):
        if (not self.mal_only and idx < len(self.choice)):
            # Return non-trojaned data
            if self.need_pad:
                # In NLP task we need to pad input with length of Troj pattern
                p_size = self.atk_setting[0]
                X, y = self.src_dataset[self.choice[idx]]
                X_padded = torch.cat([X, torch.LongTensor([0]*p_size)], dim=0)
                return X_padded, y
            else:
                return self.src_dataset[self.choice[idx]]

        if self.mal_only:
            X, y = self.src_dataset[self.mal_choice[idx]]
        else:
            X, y = self.src_dataset[self.mal_choice[idx-len(self.choice)]]
        X_new, y_new = self.troj_gen_func(X, y, self.atk_setting)
        return X_new, y_new
def generate_adversarial_examples(model, x_in, target, eps):
        attack = torchattacks.FGSM(model, eps=eps)
        perturbed_data = attack(x_in, target)
        return perturbed_data
        
def adv_train_model(model, dataloader, epoch_num, is_binary, verbose=True):
    model.train()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    eps = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    for epoch in range(epoch_num):
        cum_loss = 0.0
        cum_acc = 0.0
        tot = 0.0
        for i,(x_in, y_in) in enumerate(dataloader):
            B = x_in.size()[0]
            adv_inputs = generate_adversarial_examples(model, x_in, y_in, eps)
            pred = model(adv_inputs)
            loss = model.loss(pred, y_in)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cum_loss += loss.item() * B
            if is_binary:
                cum_acc += ((pred>0).cpu().long().eq(y_in)).sum().item()
            else:
                pred_c = pred.max(1)[1].cpu()
                cum_acc += pred_c.eq(y_in).sum().item()
            tot += B
        if verbose:
            print("Epoch %d: loss %.4f, acc %.4f"%(epoch, cum_loss/tot, cum_acc/tot))
    return




def train_model(model, dataloader, epoch_num, is_binary, verbose=True):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epoch_num):
        cum_loss = 0.0
        cum_acc = 0.0
        tot = 0.0
        for i,(x_in, y_in) in enumerate(dataloader):
            x_in = x_in.cuda()
            y_in = y_in.cuda()
            B = x_in.size()[0]
            pred = model(x_in)
            loss = model.loss(pred, y_in)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cum_loss += loss.item() * B
            if is_binary:
                cum_acc += ((pred>0).cpu().long().eq(y_in)).sum().item()
            else:
                pred_c = pred.max(1)[1].cpu()
                cum_acc += (pred_c.eq(y_in.cpu())).sum().item()
            tot = tot + B
        if verbose and epoch%10==0:
            print ("Epoch %d, loss = %.4f, acc = %.4f"%(epoch, cum_loss/tot, cum_acc/tot))
    return 


def eval_model(model, dataloader, is_binary=False):
    model.eval()
    cum_acc = 0.0
    tot = 0.0
    for i,(x_in, y_in) in enumerate(dataloader):
        B = x_in.size()[0]
        pred = model(x_in)
        if is_binary:
            cum_acc += ((pred>0).cpu().long().eq(y_in)).sum().item()
        else:
            pred_c = pred.max(1)[1].cpu()
            cum_acc += (pred_c.eq(y_in)).sum().item()
        tot = tot + B
    return cum_acc / tot
def log(message, print_to_console=True, log_level=logging.DEBUG):
    if log_level == logging.INFO:
        logging.info(message)
    elif log_level == logging.DEBUG:
        logging.debug(message)
    elif log_level == logging.WARNING:
        logging.warning(message)
    elif log_level == logging.ERROR:
        logging.error(message)
    elif log_level == logging.CRITICAL:
        logging.critical(message)
    else:
        logging.debug(message)

    if print_to_console:
        print(message)
# one-hot the layer index
def one_hot(data, max_value):
    ones = torch.sparse.torch.eye(max_value)
    return ones.index_select(0, data)
def compute_offset(task):
    offset1 = task * 0
    offset2 = (task + 1) * 0
    return int(offset1), int(offset2)
