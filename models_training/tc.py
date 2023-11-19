import json
import os
import shutil
import time
# from time import time
import torchvision.transforms as transforms
import random 
import sys, os
sys.path.append('.')
import kornia.augmentation as A
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from model_lib.preact_resnet import PreActResNet18
from model_lib.vgg import VGG
from model_lib.lenet import LeNet
from model_lib.resnet import ResNet18
from model_lib.models import Denormalizer, NetC_MNIST, Normalizer
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomErasing
import argparse
import torchattacks
import pickle
# np.random.seed(42)


_, term_width = os.popen("stty size", "r").read().split()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 65.0
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [")
    for i in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    if msg:
        L.append(" | " + msg)

    msg = "".join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()
class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):  # , **kwargs):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x

class PostTensorTransform(torch.nn.Module):
    def __init__(self, opt):
        super(PostTensorTransform, self).__init__()
        self.random_crop = ProbTransform(
            A.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop), p=0.8
        )
        self.random_rotation = ProbTransform(A.RandomRotation(opt.random_rotation), p=0.5)
        if opt.dataset == "cifar10":
            self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

def get_arguments():
    parser = argparse.ArgumentParser()


    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--continue_training", action="store_true")

    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--type_model", type=str, default='shadow')
    # parser.add_argument("--attack_mode", type=str, default="all2one")

    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr_C", type=float, default=1e-2)
    parser.add_argument("--schedulerC_milestones", type=list, default=[100, 200, 300, 400])
    parser.add_argument("--schedulerC_lambda", type=float, default=0.1)
    parser.add_argument("--n_iters", type=int, default=100)
    parser.add_argument("--num_workers", type=float, default=6)

    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--pc", type=float, default=0.1)
    parser.add_argument("--cross_ratio", type=float, default=2)  # rho_a = pc, rho_n = pc * cross_ratio

    parser.add_argument("--random_rotation", type=int, default=10)
    parser.add_argument("--random_crop", type=int, default=5)

    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--grid-rescale", type=float, default=1
    )  # scale grid values to avoid pixel values going out of [-1, 1]. For example, grid-rescale = 0.98

    return parser

def get_model(opt):
    netC = None
    optimizerC = None
    schedulerC = None

    if opt.dataset == "gtsrb" or opt.dataset == "cifar10":
        netC = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
        # netC = LeNet().to(opt.device)
        # netC = ResNet18().to(opt.device)
    if opt.dataset == "celeba":
        netC = ResNet18().to(opt.device)
    if opt.dataset == "mnist":
        netC = NetC_MNIST().to(opt.device)

    # Optimizer
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)

    # Scheduler
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)

    return netC, optimizerC, schedulerC
def generate_adversarial_examples(model, x_in, target):
        attack = torchattacks.FGSM(model, eps=0.01)
        perturbed_data = attack(x_in, target)
        return perturbed_data

def adv_train(i, netC, optimizerC, schedulerC, train_dl, opt):
    print(" Train:")
    netC.train()
  
    total_loss_ce = 0
    total_sample = 0

    total_clean = 0
    total_cross = 0
    total_clean_correct = 0
    total_cross_correct = 0
    criterion_CE = torch.nn.CrossEntropyLoss()
    criterion_BCE = torch.nn.BCELoss()

    denormalizer = Denormalizer(opt)
    transforms = PostTensorTransform(opt).to(opt.device)
    total_time = 0

    avg_acc_cross = 0

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizerC.zero_grad()

        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        adv_inputs = generate_adversarial_examples(netC, inputs, targets)
        bs = inputs.shape[0]
        start = time.time()
        total_preds = netC(adv_inputs)
        total_time += time.time() - start

        loss_ce = netC.loss(total_preds, targets)

        loss = loss_ce
        loss_ce.backward()

        optimizerC.step()

        total_sample += bs
        total_loss_ce += loss_ce

        total_clean += bs 
        # total_clean_correct += torch.sum(torch.argmax(total_preds, 1) == targets)
        pred_c = total_preds.max(1)[1]
        total_clean_correct += (pred_c.eq(targets)).sum().item()

        avg_acc_clean = total_clean_correct/ total_sample

        avg_loss_ce = total_loss_ce / total_sample


        progress_bar(
        batch_idx,
        len(train_dl),
        "CE Loss: {:.4f} | Clean Acc: {:.4f} ".format(avg_loss_ce, avg_acc_clean),
        )


    schedulerC.step()

def train(i, netC, optimizerC, schedulerC, train_dl, opt):
    print(" Train:")
    netC.train()
  
    total_loss_ce = 0
    total_sample = 0

    total_clean = 0
    # total_cross = 0
    total_clean_correct = 0
    # total_cross_correct = 0
    # criterion_CE = torch.nn.CrossEntropyLoss()
    # criterion_BCE = torch.nn.BCELoss()

    # denormalizer = Denormalizer(opt)
    # transforms = PostTensorTransform(opt).to(opt.device)
    total_time = 0

    # avg_acc_cross = 0

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizerC.zero_grad()

        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]
        start = time.time()
        total_preds = netC(inputs)
        total_time += time.time() - start

        loss_ce = netC.loss(total_preds, targets)

        loss = loss_ce
        loss_ce.backward()

        optimizerC.step()

        total_sample += bs
        total_loss_ce += loss_ce

        total_clean += bs 
        # total_clean_correct += torch.sum(torch.argmax(total_preds, 1) == targets)
        pred_c = total_preds.max(1)[1]
        total_clean_correct += (pred_c.eq(targets)).sum().item()

        avg_acc_clean = total_clean_correct/ total_sample

        avg_loss_ce = total_loss_ce / total_sample


        progress_bar(
        batch_idx,
        len(train_dl),
        "CE Loss: {:.4f} | Clean Acc: {:.4f} ".format(avg_loss_ce, avg_acc_clean),
        )
    schedulerC.step()


def eval(i,
    netC,
    optimizerC,
    schedulerC,
    test_dl,
    best_clean_acc,
    best_bd_acc,
    best_cross_acc,
    opt,
):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0

    total_ae_loss = 0

    criterion_BCE = torch.nn.BCELoss()

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            preds_clean = netC(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            acc_clean = total_clean_correct * 100.0 / total_sample


        info_string = "Acc: {:.4f} ".format(
                    acc_clean
                )
        progress_bar(batch_idx, len(test_dl), info_string)

    


        if opt.type_model == 'shadow':
            save_path = './Ablation/%s'%opt.dataset+'/fraction/train_benign0.5_%d.model'%i
        else:
            save_path = './test/%s'%opt.dataset+'/models/target_benign_%d.model'%i
        torch.save(netC.state_dict(), save_path)
        print ("Benign model saved to %s"%save_path)


    return best_clean_acc, best_bd_acc, best_cross_acc


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    # if GPU:
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    opt = get_arguments().parse_args()

    if opt.type_model == 'shadow':
        NUM = 5
    else:
        NUM = 10

    if opt.dataset in ["mnist", "cifar10"]:
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./raw_data/', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./raw_data/', train=False, download=True, transform=transform)
        # balanced_batch_sampler = BalancedBatchSampler(trainset, 10, 10)
        tot_num = len(trainset)
        if opt.type_model == 'shadow':
            shadow_indices = np.random.choice(tot_num, int(tot_num*0.5))
        else:
            shadow_indices = np.random.choice(tot_num, int(tot_num*1))
        shadow_set = torch.utils.data.Subset(trainset, shadow_indices)
        train_dl = torch.utils.data.DataLoader(shadow_set, batch_size=100, shuffle=True, num_workers=4)
        test_dl = torch.utils.data.DataLoader(testset, batch_size=100, num_workers=2)
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3

    elif opt.dataset == "gtsrb":
        print("Loading GTSRB dataset")
        transform = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor()
            ])
        trainset = torchvision.datasets.GTSRB(root='./raw_data/', split='train', download=True, transform=transform)
        testset = torchvision.datasets.GTSRB(root='./raw_data/', split='test', download=True, transform=transform)
        # balanced_batch_sampler = BalancedBatchSampler(trainset, 43, 2)
        tot_num = len(trainset)
        if opt.type_model == 'shadow':
            shadow_indices = np.random.choice(tot_num, int(tot_num*0.1))
        else:
            shadow_indices = np.random.choice(tot_num, int(tot_num*1))
        print("shadow_indices", len(shadow_indices))
        shadow_set = torch.utils.data.Subset(trainset, shadow_indices)
        train_dl = torch.utils.data.DataLoader(shadow_set, batch_size=100, num_workers=2)
        test_dl = torch.utils.data.DataLoader(testset, batch_size=100, num_workers=2)
                # print("shadow_indices", len(shadow_indices))
        # adv_indices = np.random.choice(len(trainset), int(len(trainset)*0.2), replace=False)
        # adv_set = torch.utils.data.Subset(trainset, shadow_indices)
        adv_dl = torch.utils.data.DataLoader(shadow_set, batch_size=100, shuffle=True, num_workers=2)
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")
    # pickle the shadow dataset for later use
    shadow_set_path = './test/%s'%opt.dataset+'/shadow_set0.5.pkl'
    with open(shadow_set_path, 'wb') as f:
        pickle.dump(shadow_set, f)

    for i in range(NUM):
        # prepare model
        netC, optimizerC, schedulerC = get_model(opt)
        print("Train from scratch!!!")
        best_clean_acc = 0.0
        best_bd_acc = 0.0
        best_cross_acc = 0.0
        epoch_current = 0
        # for epoch in range(epoch_current, opt.n_iters):
        #         print("Epoch {}:".format(epoch + 1))
        #         adv_train(i, netC, optimizerC, schedulerC, adv_dl, opt)

        # Calculate the time needed for training
        start_time = time.time()
        for epoch in range(epoch_current, opt.n_iters):
                print("Epoch {}:".format(epoch + 1))
                train(i, netC, optimizerC, schedulerC, train_dl, opt)
                best_clean_acc, best_bd_acc, best_cross_acc = eval(i,
                netC,
                optimizerC,
                schedulerC,
                test_dl,
                best_clean_acc,
                best_bd_acc,
                best_cross_acc,

                opt,
                )
        end_time = time.time()
        print("Time needed for training: {:.2f} seconds".format(end_time - start_time))
if __name__ == "__main__":
    main()
        