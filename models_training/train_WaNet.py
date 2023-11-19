import json
import os
import shutil
import time
# from time import time
import sys, os
sys.path.append('.')
import torchvision.transforms as transforms
import random 
import sys
import kornia.augmentation as A
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from model_lib.preact_resnet import PreActResNet18
from model_lib.vgg import VGG
from model_lib.lenet import LeNet
from model_lib.chest_cnn_model import Model
from torchvision import transforms, datasets, models


from model_lib.resnet import ResNet18
from model_lib.models import Denormalizer, NetC_MNIST, Normalizer
from torch import nn
# from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomErasing
import argparse
from IA.wanet_dataloader import get_WaNetdataloader



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

    parser.add_argument("--data_root", type=str, default="/home/ubuntu/temps/")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints")
    parser.add_argument("--temps", type=str, default="./temps")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--continue_training", action="store_true")

    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--attack_mode", type=str, default="all2one")

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

    parser.add_argument("--s", type=float, default=1)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--grid-rescale", type=float, default=1
    )  # scale grid values to avoid pixel values going out of [-1, 1]. For example, grid-rescale = 0.98

    return parser

def get_model(opt):
    netC = None
    optimizerC = None
    schedulerC = None

    if opt.dataset == "gtsrb":
        # netC = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
        # netC = models.resnet18(pretrained=False)
        # num_ftrs = netC.fc.in_features
        # netC.fc = nn.Linear(num_ftrs, 43)
        netC = models.vgg16(pretrained=False)
        num_ftrs = netC.classifier[6].in_features
        netC.classifier[6] = nn.Linear(num_ftrs, 43)
        netC = netC.to(opt.device)
    if opt.dataset == "cifar10":
        # netC = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
        # netC = models.resnet18(pretrained=False)
        # num_ftrs = netC.fc.in_features
        # netC.fc = nn.Linear(num_ftrs, 10)
        netC = models.vgg16(pretrained=False)
        num_ftrs = netC.classifier[6].in_features
        netC.classifier[6] = nn.Linear(num_ftrs, 10)
        netC = netC.to(opt.device)
    if opt.dataset == "celeba":
        netC = ResNet18().to(opt.device)

    if opt.dataset == "mnist":
        netC = NetC_MNIST().to(opt.device)
    if opt.dataset == "chest":
        netC = Model(gpu=True)

    # Optimizer
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)

    # Scheduler
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)

    return netC, optimizerC, schedulerC


def train(netC, optimizerC, schedulerC, train_dl, noise_grid, identity_grid, epoch, opt):
    print(" Train:")
    netC.train()
    rate_bd = opt.pc
    total_loss_ce = 0
    total_sample = 0

    total_clean = 0
    total_bd = 0
    total_cross = 0
    total_clean_correct = 0
    total_bd_correct = 0
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

        # Create backdoor data
        num_bd = int(bs * rate_bd)
        num_cross = int(num_bd * opt.cross_ratio)
        grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        ins = torch.rand(num_cross, opt.input_height, opt.input_height, 2).to(opt.device) * 2 - 1
        grid_temps2 = grid_temps.repeat(num_cross, 1, 1, 1) + ins / opt.input_height
        grid_temps2 = torch.clamp(grid_temps2, -1, 1)

        inputs_bd = F.grid_sample(inputs[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
        if opt.attack_mode == "all2one":
            targets_bd = torch.ones_like(targets[:num_bd]) * opt.target_label
        if opt.attack_mode == "all2all":
            targets_bd = torch.remainder(targets[:num_bd] + 1, opt.num_classes)

        inputs_cross = F.grid_sample(inputs[num_bd : (num_bd + num_cross)], grid_temps2, align_corners=True)

        total_inputs = torch.cat([inputs_bd, inputs_cross, inputs[(num_bd + num_cross) :]], dim=0)
        total_inputs = transforms(total_inputs)
        total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)
        start = time.time()
        total_preds = netC(total_inputs)
        total_time += time.time() - start

        loss_ce = criterion_CE(total_preds, total_targets)

        loss = loss_ce
        loss.backward()

        optimizerC.step()

        total_sample += bs
        total_loss_ce += loss_ce.detach()

        total_clean += bs - num_bd - num_cross
        total_bd += num_bd
        total_cross += num_cross
        total_clean_correct += torch.sum(
            torch.argmax(total_preds[(num_bd + num_cross) :], dim=1) == total_targets[(num_bd + num_cross) :]
        )
        total_bd_correct += torch.sum(torch.argmax(total_preds[:num_bd], dim=1) == targets_bd)
        if num_cross:
            total_cross_correct += torch.sum(
                torch.argmax(total_preds[num_bd : (num_bd + num_cross)], dim=1)
                == total_targets[num_bd : (num_bd + num_cross)]
            )
            avg_acc_cross = total_cross_correct * 100.0 / total_cross

        avg_acc_clean = total_clean_correct * 100.0 / total_clean
        avg_acc_bd = total_bd_correct * 100.0 / total_bd

        avg_loss_ce = total_loss_ce / total_sample

        if num_cross:
            progress_bar(
                batch_idx,
                len(train_dl),
                "CE Loss: {:.4f} | Clean Acc: {:.4f} | Bd Acc: {:.4f} | Cross Acc: {:.4f}".format(
                    avg_loss_ce, avg_acc_clean, avg_acc_bd, avg_acc_cross
                ),
            )
        else:
            progress_bar(
                batch_idx,
                len(train_dl),
                "CE Loss: {:.4f} | Clean Acc: {:.4f} | Bd Acc: {:.4f} ".format(avg_loss_ce, avg_acc_clean, avg_acc_bd),
            )

        # # Save image for debugging
        # if not batch_idx % 50:
        #     if not os.path.exists(opt.temps):
        #         os.makedirs(opt.temps)
        #     path = os.path.join(opt.temps, "backdoor_image.png")
        #     torchvision.utils.save_image(inputs_bd, path, normalize=True)

    schedulerC.step()


def eval(i,
    netC,
    optimizerC,
    schedulerC,
    test_dl,
    noise_grid,
    identity_grid,
    best_clean_acc,
    best_bd_acc,
    best_cross_acc,
    epoch,
    opt,
):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0
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

            # Evaluate Backdoor
            grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)

            ins = torch.rand(bs, opt.input_height, opt.input_height, 2).to(opt.device) * 2 - 1
            grid_temps2 = grid_temps.repeat(bs, 1, 1, 1) + ins / opt.input_height
            grid_temps2 = torch.clamp(grid_temps2, -1, 1)

            inputs_bd = F.grid_sample(inputs, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets) * opt.target_label
            if opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets + 1, opt.num_classes)
            preds_bd = netC(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            acc_clean = total_clean_correct * 100.0 / total_sample
            acc_bd = total_bd_correct * 100.0 / total_sample

            # Evaluate cross
            if opt.cross_ratio:
                inputs_cross = F.grid_sample(inputs, grid_temps2, align_corners=True)
                preds_cross = netC(inputs_cross)
                total_cross_correct += torch.sum(torch.argmax(preds_cross, 1) == targets)

                acc_cross = total_cross_correct * 100.0 / total_sample

                info_string = (
                    "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f} | Cross: {:.4f}".format(
                        acc_clean, best_clean_acc, acc_bd, best_bd_acc, acc_cross, best_cross_acc
                    )
                )
            else:
                info_string = "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f}".format(
                    acc_clean, best_clean_acc, acc_bd, best_bd_acc
                )
            progress_bar(batch_idx, len(test_dl), info_string)

    # tensorboard
    # if not epoch % 1:
    #     tf_writer.add_scalars("Test Accuracy", {"Clean": acc_clean, "Bd": acc_bd}, epoch)

    # Save checkpoint
    if acc_clean > best_clean_acc or (acc_clean > best_clean_acc - 0.1 and acc_bd > best_bd_acc):
        print(" Saving...")
        best_clean_acc = acc_clean
        best_bd_acc = acc_bd
        if opt.cross_ratio:
            best_cross_acc = acc_cross
        else:
            best_cross_acc = torch.tensor([0])
        state_dict = {
            "netC": netC.state_dict(),
            "schedulerC": schedulerC.state_dict(),
            "optimizerC": optimizerC.state_dict(),
            "best_clean_acc": best_clean_acc,
            "best_bd_acc": best_bd_acc,
            "best_cross_acc": best_cross_acc,
            "epoch_current": epoch,
            "identity_grid": identity_grid,
            "noise_grid": noise_grid,
        }
        save_path = './general_train_model_tae_ckpt/%s'%opt.dataset+'/models/target_trojaned_WaNet_vgg_%d.model'%i
        torch.save(netC.state_dict(), save_path)
        print ("WaNet model saved to %s"%save_path)


    return best_clean_acc, best_bd_acc, best_cross_acc


def main():
    for i in range(20):
        opt = get_arguments().parse_args()

        if opt.dataset in ["mnist", "cifar10"]:
            opt.num_classes = 10
        elif opt.dataset == "gtsrb":
            opt.num_classes = 43
        elif opt.dataset == "chest":
            opt.num_classes = 2
        else:
            raise Exception("Invalid Dataset")

        if opt.dataset == "cifar10":
            opt.input_height = 32
            opt.input_width = 32
            opt.input_channel = 3
        elif opt.dataset == "chest":
            opt.input_height = 32
            opt.input_width = 32
            opt.input_channel = 1

        elif opt.dataset == "gtsrb":
            opt.input_height = 32
            opt.input_width = 32
            opt.input_channel = 3
        elif opt.dataset == "mnist":
            opt.input_height = 28
            opt.input_width = 28
            opt.input_channel = 1
        else:
            raise Exception("Invalid Dataset")

        # Dataset
        train_dl = get_WaNetdataloader(opt, True)
        test_dl = get_WaNetdataloader(opt, False)

        # prepare model
        netC, optimizerC, schedulerC = get_model(opt)

        # Load pretrained model
        mode = opt.attack_mode
        opt.ckpt_folder = os.path.join(opt.checkpoints, opt.dataset)
        opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_morph.pth.tar".format(opt.dataset, mode))
        opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")
        if not os.path.exists(opt.log_dir):
            os.makedirs(opt.log_dir)

        if opt.continue_training:
            if os.path.exists(opt.ckpt_path):
                print("Continue training!!")
                state_dict = torch.load(opt.ckpt_path)
                netC.load_state_dict(state_dict["netC"])
                optimizerC.load_state_dict(state_dict["optimizerC"])
                schedulerC.load_state_dict(state_dict["schedulerC"])
                best_clean_acc = state_dict["best_clean_acc"]
                best_bd_acc = state_dict["best_bd_acc"]
                best_cross_acc = state_dict["best_cross_acc"]
                epoch_current = state_dict["epoch_current"]
                identity_grid = state_dict["identity_grid"]
                noise_grid = state_dict["noise_grid"]
                # tf_writer = SummaryWriter(log_dir=opt.log_dir)
            else:
                print("Pretrained model doesnt exist")
                exit()
        else:
            print("Train from scratch!!!")
            best_clean_acc = 0.0
            best_bd_acc = 0.0
            best_cross_acc = 0.0
            epoch_current = 0

            # Prepare grid
            ins = torch.rand(1, 2, opt.k, opt.k) * 2 - 1
            ins = ins / torch.mean(torch.abs(ins))
            noise_grid = (
                F.upsample(ins, size=opt.input_height, mode="bicubic", align_corners=True)
                .permute(0, 2, 3, 1)
                .to(opt.device)
            )
            array1d = torch.linspace(-1, 1, steps=opt.input_height)
            x, y = torch.meshgrid(array1d, array1d)
            identity_grid = torch.stack((y, x), 2)[None, ...].to(opt.device)

            shutil.rmtree(opt.ckpt_folder, ignore_errors=True)
            # os.makedirs(opt.log_dir)
            # with open(os.path.join(opt.ckpt_folder, "opt.json"), "w+") as f:
            #     json.dump(opt.__dict__, f, indent=2)
            # tf_writer = SummaryWriter(log_dir=opt.log_dir)

        for epoch in range(epoch_current, opt.n_iters):
            print("Epoch {}:".format(epoch + 1))
            train(netC, optimizerC, schedulerC, train_dl, noise_grid, identity_grid, epoch, opt)
            best_clean_acc, best_bd_acc, best_cross_acc = eval(i,
                netC,
                optimizerC,
                schedulerC,
                test_dl,
                noise_grid,
                identity_grid,
                best_clean_acc,
                best_bd_acc,
                best_cross_acc,
                epoch,
                opt,
            )


if __name__ == "__main__":
    main()