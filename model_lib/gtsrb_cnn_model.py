import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, gpu=False):
        super().__init__()
        self.gpu = gpu
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*7*7, 1000),
            nn.ReLU(inplace=True),
            
            nn.Dropout(0.5),
            nn.Linear(in_features=1000, out_features=256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, 43)
            )
        if self.gpu:
            self.cuda()

    def forward(self, x):
        x = x.cuda() if self.gpu else x
        x = self.features(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)

# class Model(nn.Module):
#     def __init__(self, gpu=False):
#         super(Model, self).__init__()
#         self.gpu = gpu
#         # CNN layers
#         self.conv1 = nn.Conv2d(3, 100, kernel_size=5)
#         self.bn1 = nn.BatchNorm2d(100)
#         self.conv2 = nn.Conv2d(100, 150, kernel_size=3)
#         self.bn2 = nn.BatchNorm2d(150)
#         self.conv3 = nn.Conv2d(150, 250, kernel_size=3)
#         self.bn3 = nn.BatchNorm2d(250)
#         self.conv_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(250*2*2, 350)
#         self.fc2 = nn.Linear(350, 43)

#         self.localization = nn.Sequential(
#             nn.Conv2d(3, 8, kernel_size=7),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#             nn.Conv2d(8, 10, kernel_size=5),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True)
#             )

#         # Regressor for the 3 * 2 affine matrix
#         self.fc_loc = nn.Sequential(
#             nn.Linear(10 * 4 * 4, 32),
#             nn.ReLU(True),
#             nn.Linear(32, 3 * 2)
#             )
   
#         # Initialize the weights/bias with identity transformation
#         self.fc_loc[2].weight.data.zero_()
#         self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

#         if(self.gpu):
#             self.cuda()
#     # Spatial transformer network forward function
#     def stn(self, x):
#         xs = self.localization(x)
#         xs = xs.view(-1, 10 * 4 * 4)
#         theta = self.fc_loc(xs)
#         theta = theta.view(-1, 2, 3)
#         grid = F.affine_grid(theta, x.size())
#         x = F.grid_sample(x, grid)
#         return x

#     def forward(self, x):
#         # transform the input
#         x = x.cuda()
#         x = self.stn(x)

#         # Perform forward pass
#         x = self.bn1(F.max_pool2d(F.leaky_relu(self.conv1(x)),2))
#         x = self.conv_drop(x)
#         x = self.bn2(F.max_pool2d(F.leaky_relu(self.conv2(x)),2))
#         x = self.conv_drop(x)
#         x = self.bn3(F.max_pool2d(F.leaky_relu(self.conv3(x)),2))
#         x = self.conv_drop(x)
#         x = x.view(-1, 250*2*2)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return x, None

#     def loss(self, pred, label):
#         if self.gpu:
#             label = label.cuda()
#         return F.cross_entropy(pred, label)


def random_troj_setting(troj_type):
    print("generating random trojan settings")
    MAX_SIZE = 112
    CLASS_NUM = 43

    if troj_type == 'jumbo':
        p_size = np.random.choice([2,3,4,5,MAX_SIZE], 1)[0]
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
    print("original label: ", y)
    print("new label: ", y_new)
    return X_new, y_new


