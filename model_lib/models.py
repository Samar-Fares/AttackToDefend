import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.nn import Module
from torchvision import transforms

from .blocks import *


class Normalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = (x[:, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone


class Denormalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = x[:, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone


class Normalizer:
    def __init__(self, opt):
        self.normalizer = self._get_normalizer(opt)

    def _get_normalizer(self, opt):
        if opt.dataset == "cifar10":
            normalizer = Normalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            normalizer = Normalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
            normalizer = None
        else:
            raise Exception("Invalid dataset")
        return normalizer

    def __call__(self, x):
        if self.normalizer:
            x = self.normalizer(x)
        return x


class Denormalizer:
    def __init__(self, opt):
        self.denormalizer = self._get_denormalizer(opt)

    def _get_denormalizer(self, opt):
        if opt.dataset == "cifar10":
            denormalizer = Denormalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            denormalizer = Denormalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb" or opt.dataset == "celeba" or opt.dataset == "chest":
            denormalizer = None
        else:
            raise Exception("Invalid dataset")
        return denormalizer

    def __call__(self, x):
        if self.denormalizer:
            x = self.denormalizer(x)
        return x


# ---------------------------- Classifiers ----------------------------#
class MNISTBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(MNISTBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.ind = None

    def forward(self, x):
        return self.conv1(F.relu(self.bn1(x)))


        
class Generator(nn.Sequential):
    def __init__(self, opt, out_channels = None):
        super(Generator, self).__init__()
        if(opt.dataset == 'mnist'):
            channel_init = 16
            steps = 2
        else:
            channel_init = 32
            steps = 3

        channel_current = opt.input_channel
        channel_next = channel_init
        for step in range(steps):
            self.add_module('convblock_down_{}'.format(2 * step), Conv2dBlock(channel_current, channel_next))
            self.add_module('convblock_down_{}'.format(2 * step + 1), Conv2dBlock(channel_next, channel_next))
            self.add_module('downsample_{}'.format(step), DownSampleBlock())
            if(step < steps - 1):
                channel_current = channel_next
                channel_next *= 2

        self.add_module('convblock_middle', Conv2dBlock(channel_next, channel_next))

        channel_current = channel_next
        channel_next = channel_current // 2
        for step in range(steps):
            self.add_module('upsample_{}'.format(step), UpSampleBlock())
            self.add_module('convblock_up_{}'.format(2 * step), Conv2dBlock(channel_current, channel_current))
            if step == steps - 1:
                self.add_module('convblock_up_{}'.format(2 * step + 1), Conv2dBlock(channel_current, channel_next, relu=False))
            else:
                self.add_module('convblock_up_{}'.format(2 * step + 1), Conv2dBlock(channel_current, channel_next))
            channel_current = channel_next
            channel_next = channel_next // 2
            if(step == steps - 2):
                if out_channels is None:
                   channel_next = opt.input_channel
                else:
                   channel_next = out_channels          

        self._EPSILON = 1e-7
        self._normalizer = self._get_normalize(opt)
        self._denormalizer = self._get_denormalize(opt)

    def _get_denormalize(self, opt):
        if(opt.dataset == 'cifar10'):
            denormalizer = Denormalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif(opt.dataset == 'mnist'):
            denormalizer = Denormalize(opt, [0.5], [0.5])
        elif(opt.dataset == 'gtsrb') or (opt.dataset == 'chest'):
            denormalizer = None
        else: 
            raise Exception("Invalid dataset")
        return denormalizer

    def _get_normalize(self, opt):
        if(opt.dataset == 'cifar10'):
            normalizer = Normalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif(opt.dataset == 'mnist'):
            normalizer = Normalize(opt, [0.5], [0.5])
        elif(opt.dataset == 'gtsrb') or (opt.dataset == 'chest'):
            normalizer = None
        else:
            raise Exception("Invalid dataset")
        return normalizer

    def forward(self, x):
        for module in self.children():
            x = module(x)
        x = nn.Tanh()(x) / (2 + self._EPSILON) + 0.5
        return x

    def normalize_pattern(self, x):
        if(self._normalizer):
            x = self._normalizer(x)
        return x

    def denormalize_pattern(self, x):
        if(self._denormalizer):
            x = self._denormalizer(x)
        return x 

    def threshold(self, x):
        return nn.Tanh()(x*20 - 10) / (2 + self._EPSILON) + 0.5

class NetC_MNIST(nn.Module):
    def __init__(self):
        super(NetC_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), 2, 1)  # 14
        self.relu1 = nn.ReLU(inplace=True)
        self.layer2 = MNISTBlock(32, 64, 2)  # 7
        self.layer3 = MNISTBlock(64, 64, 2)  # 4
        self.flatten = nn.Flatten()
        self.linear6 = nn.Linear(64 * 4 * 4, 512)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout8 = nn.Dropout(0.3)
        self.linear9 = nn.Linear(512, 10)
        self.cuda()

    def forward(self, x):
        x = x.cuda()
        for module in self.children():
            x = module(x)
        return x
    def loss(self, x, y):
        return F.cross_entropy(x, y)