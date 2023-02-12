import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.alexnet import AlexNet
import torchvision
from torchvision.models.resnet import ResNet, BasicBlock

# class Model(ResNet):
#     def __init__(self, gpu=True):
#         super(Model, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
#         self.gpu = gpu
#         self.model = torchvision.models.resnet18(pretrained=True)
#         # self.model.conv1 = torch.nn.Conv2d(1, 64, 
#         #     kernel_size=(7, 7), 
#         #     stride=(2, 2), 
#         #     padding=(3, 3), bias=False)

#         if gpu:
#             self.cuda()

#     def forward(self, x):
#         if self.gpu:
#             x = x.cuda()
#         return self.model(x), None

#     def loss(self, pred, label):
#         if self.gpu:
#             label = label.cuda()
#         return F.cross_entropy(pred, label)
# class Model(nn.Module):
#     def __init__(self, output_dim = 10, gpu=True):
#         super().__init__()
#         self.gpu = gpu
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, 3, 2, 1),  # in_channels, out_channels, kernel_size, stride, padding
#             nn.MaxPool2d(2),  # kernel_size
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 192, 3, padding=1),
#             nn.MaxPool2d(2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(192, 384, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, 3, padding=1),
#             nn.MaxPool2d(2),
#             nn.ReLU(inplace=True)
#         )

#         self.classifier = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(256 * 2 * 2, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, output_dim),
#         )
#         if gpu:
#             self.cuda()

#     def forward(self, x):
#         if self.gpu:
#             x = x.cuda()
#         x = self.features(x)
#         h = x.view(x.shape[0], -1)
#         x = self.classifier(h)
#         return x, h

    # def loss(self, pred, label):
    #     if self.gpu:
    #         label = label.cuda()
    #     return F.cross_entropy(pred, label)
class Model(nn.Module):
    def __init__(self, gpu=False):
        super(Model, self).__init__()
        self.gpu = gpu

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(64*8*8, 256)
        self.fc = nn.Linear(256, 256)
        self.output = nn.Linear(256, 10)

        if gpu:
            self.cuda()

    def forward(self, x):
        out = {}
        if self.gpu:
            x = x.cuda()
        B = x.size()[0]
        x = self.conv1(x)
        # out['conv1'] = x
        x = F.relu(x)
        out['relu1'] = x
        x = self.conv2(x)
        # out['conv2'] = x
        x = F.relu(x)
        out['relu2'] = x
        x = self.max_pool(x)
        x = self.conv3(x)
        # out['conv3'] = x
        x = F.relu(x)
        out['relu3'] = x
        x = self.conv4(x)
        # out['conv4'] = x
        x = F.relu(x)
        out['relu4'] = x
        x = self.max_pool(x)
        x = self.linear(x.view(B,64*8*8))
        out['linear'] = x
        x = F.relu(x)
        out['relu5'] = x
        x = self.fc(x)
        out['fc'] = x
        x = F.relu(x)
        out['relu6'] = x
        x = F.dropout(x, 0.5, training=self.training)
        x = self.output(x)
        out['output'] = x

        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)

# def random_troj_setting(troj_type):
#     MAX_SIZE = 32
#     CLASS_NUM = 10

#     if troj_type == 'jumbo':
#         p_size = np.random.choice([2,3,4,5,MAX_SIZE], 1)[0]
#         if p_size < MAX_SIZE:
#             alpha = np.random.uniform(0.2, 0.6)
#             if alpha > 0.5:
#                 alpha = 1.0
#         else:
#             alpha = np.random.uniform(0.05, 0.2)
#     elif troj_type == 'M':
#         # p_size = np.random.choice([2,3,4,5], 1)[0]
#         p_size = 3
#         alpha = 1.0
#     elif troj_type == 'B':
#         p_size = MAX_SIZE
#         alpha = np.random.uniform(0.05, 0.2)

#     if p_size < MAX_SIZE:
#         # loc_x = np.random.randint(MAX_SIZE-p_size)
#         loc_x = 15
#         # loc_y = np.random.randint(MAX_SIZE-p_size)
#         loc_y = 10
#         loc = (loc_x, loc_y)
#     else:
#         loc = (0, 0)

#     # eps = np.random.uniform(0, 1)
#     eps = 0.7
#     # pattern = np.random.uniform(-eps, 1+eps,size=(3,p_size,p_size))
#     pattern = [[[ 1.00455709,  1.3414555,  -0.20180177],
#                 [ 1.18995585, -0.60580244,  0.93466984],
#                 [ 1.42193991,  1.26601813,  1.14994004]],

#                 [[ 1.10517016, -0.47931221, -0.23478849],
#                 [ 1.10583767, -0.62343919, -0.42594646],
#                 [-0.69768964, -0.37021179,  0.2584735 ]],

#                 [[ 1.64407929,  0.85235963,  0.1980018 ],
#                 [ 0.69938451, -0.36851951,  1.20650072],
#                 [ 0.82711391,  0.56972092,  0.26023445]]]
#     pattern = np.clip(pattern,0,1)
#     # target_y = np.random.randint(CLASS_NUM)
#     target_y = 1
#     # inject_p = np.random.uniform(0.05, 0.5)
#     inject_p = 0.3

#     return p_size, pattern, loc, alpha, target_y, inject_p
"""VGG11/13/16/19 in Pytorch."""
import torch
import torch.nn as nn


cfg = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, gpu=False):
        super(VGG, self).__init__()
        self.gpu = gpu
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        
        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


# def test():
#     net = VGG("VGG11")
#     x = torch.randn(2, 3, 32, 32)
#     y = net(x)
#     print(y.size())


# test()

"""LeNet in PyTorch."""
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, gpu=False):
        super(LeNet, self).__init__()
        self.gpu = gpu
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)
        
def random_troj_setting(troj_type):
    MAX_SIZE = 32
    CLASS_NUM = 10

    if troj_type == 'jumbo':
        p_size = np.random.choice([2,3,4,5,MAX_SIZE], 1)[0]
    elif troj_type == 'M':
        p_size = np.random.choice([2,3,4,5], 1)[0]
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
    print('original label: ', y, 'target label: ', y_new, 'pattern size: ', p_size, 'pattern location: ', loc, 'alpha: ', alpha, 'inject probability: ', inject_p)
    return X_new, y_new






 
# class Model(nn.Module):
#     def __init__(self, gpu=False):
#         super().__init__()
#         self.gpu = gpu
#         self.network = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2), # output: 128 x 8 x 8
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

#             nn.Flatten(), 
#             nn.Linear(256*4*4, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10))
#         if gpu:
#             self.cuda()
#     def forward(self, xb):
#         if self.gpu:
#             xb = xb.cuda()
#         return self.network(xb), 'none'
#     def loss(self, pred, label):
#         if self.gpu:
#             label = label.cuda()
#         return F.cross_entropy(pred, label)

