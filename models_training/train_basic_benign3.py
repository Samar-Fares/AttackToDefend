
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
from model_lib.resnet18 import ResNet
from model_lib.resnet import ResNet18
from model_lib.models import Denormalizer
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomErasing
import argparse
import torchattacks

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

def random_troj_setting(troj_type):
    print("generating random trojan settings")
    MAX_SIZE = 32
    CLASS_NUM = 10 # 10 for CIFAR10, 43 for GTSRB

    if troj_type == 'jumbo':
        p_size = np.random.choice([6,7,8,9,MAX_SIZE], 1)[0]
    elif troj_type == 'M':
        p_size = np.random.choice([6,7,8,9], 1)[0]
    elif troj_type == 'B':
        p_size = MAX_SIZE

    alpha = np.random.uniform(0.05, 0.2)

    if p_size < MAX_SIZE:
        loc_x = np.random.randint(MAX_SIZE-p_size)
        loc_y = np.random.randint(MAX_SIZE-p_size)
        loc = (loc_x, loc_y)
    else:
        loc = (0, 0)

    eps = np.random.uniform(0, 1)
    pattern = np.random.uniform(-eps, 1+eps,size=(3,p_size,p_size))
    pattern = np.clip(pattern,0,1)
    target_y = np.random.randint(CLASS_NUM)
    inject_p = np.random.uniform(0.05, 0.5)

    return p_size, pattern, loc, alpha, target_y, inject_p


def troj_gen_func(X, y, atk_setting):
    p_size, pattern, loc, alpha, target_y, inject_p = atk_setting

    w, h = loc
    X_new = X.clone()
    X_new[:, w:w+p_size, h:h+p_size] = alpha * torch.FloatTensor(pattern) + (1-alpha) * X_new[:, w:w+p_size, h:h+p_size]
    y_new = target_y
    return X_new, y_new




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
        # netC = ResNet18().to(opt.device)
    if opt.dataset == "celeba":
        netC = ResNet18().to(opt.device)

    # Optimizer
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)

    # Scheduler
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)

    return netC, optimizerC, schedulerC
def generate_adversarial_examples(model, x_in, target):
        attack = torchattacks.FGSM(model, eps=0.001)
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


        info_string = "Acc on Backdoor: {:.4f} ".format(
                    acc_clean
                )
        progress_bar(batch_idx, len(test_dl), info_string)

    


        if opt.type_model == 'shadow':
            save_path = './test/%s'%opt.dataset+'/models/train_trojaned_%d.model'%i
        else:
            save_path = './test/%s'%opt.dataset+'/models/target_trojaned_%s_%d.model'%(opt.type_model,i)
        torch.save(netC.state_dict(), save_path)
        print ("benign model saved to %s"%save_path)


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
        NUM = 10
    else:
        NUM = 10

    if opt.dataset == "cifar10":
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
        tot_num = len(trainset)
        if opt.type_model == 'shadow':
            shadow_indices = np.random.choice(tot_num, int(tot_num*0.1))
            atk_setting = random_troj_setting('jumbo')

        else:
            shadow_indices = np.random.choice(tot_num, int(tot_num*1))
            atk_setting = random_troj_setting(opt.type_model)
        trainset_mal = BackdoorDataset(trainset, atk_setting, troj_gen_func, choice=shadow_indices, need_pad=False)
        train_dl = torch.utils.data.DataLoader(trainset_mal, batch_size=100, shuffle=True, num_workers=2)
        testset_mal = BackdoorDataset(testset, atk_setting, troj_gen_func, mal_only=True)
        test_dl = torch.utils.data.DataLoader(testset_mal, batch_size=100, num_workers=2)
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
        if opt.type_model == 'shadow':
            shadow_indices = np.random.choice(len(trainset), int(len(trainset)*0.2))
            atk_setting = random_troj_setting('jumbo')
        else:
            shadow_indices = np.random.choice(len(trainset), int(len(trainset)*1))
            atk_setting = random_troj_setting(opt.type_model)
        # print("shadow_indices", len(shadow_indices))
        # adv_indices = np.random.choice(len(trainset), int(len(trainset)*0.2), replace=False)
        # adv_set = torch.utils.data.Subset(trainset, adv_indices)
        # adv_dl = torch.utils.data.DataLoader(adv_set, batch_size=100, shuffle=True, num_workers=2)

        trainset_mal = BackdoorDataset(trainset, atk_setting, troj_gen_func, choice=shadow_indices, need_pad=False)
        train_dl = torch.utils.data.DataLoader(trainset_mal, batch_size=100, shuffle=True, num_workers=2)
        testset_mal = BackdoorDataset(testset, atk_setting, troj_gen_func, mal_only=True)
        test_dl = torch.utils.data.DataLoader(testset_mal, batch_size=100, num_workers=2)
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    for i in range(NUM):


        # Dataset
        # train_dl = get_dataloader(opt, True)
        # test_dl = get_dataloader(opt, False)

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

        for epoch in range(epoch_current, opt.n_iters):
                print("Epoch {}:".format(epoch + 1))
                # adv_train(i, netC, optimizerC, schedulerC, adv_dl, opt)
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


if __name__ == "__main__":
    main()
        