import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock




class Model(nn.Module):
    def __init__(self, gpu=False):
        super(Model, self).__init__()
        self.gpu = gpu

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=0)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32*4*4, 512)
        self.output = nn.Linear(512, 10)

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        B = x.size()[0]
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.fc(x.view(B,32*4*4))
        x = F.relu(x)
        x = self.output(x)

        return x # x: B x 10, out: B x 32 x 4 x 4

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)
        

# class Model(ResNet):
#     def __init__(self, gpu=True):
#         super(Model, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
#         self.gpu = gpu
#         self.model = torchvision.models.resnet18(pretrained=False)
#         self.model.conv1 = torch.nn.Conv2d(1, 64, 
#             kernel_size=(7, 7), 
#             stride=(2, 2), 
#             padding=(3, 3), bias=False)

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

class CNNModel(nn.Module):
    """docstring for CNNModel"""
    def __init__(self, gpu=False):
        super(CNNModel, self).__init__()
        self.gpu = gpu

        # For valid padding -> padding = 0  and change sizes 
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.cnn2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        #Max/Avg  Pool 1
        self.pool1= nn.MaxPool2d(kernel_size=2)
        #self.pool1= nn.AvgPool2d(kernel_size=2)
        self.cnn3 = nn.Conv2d(in_channels=8 , out_channels=16, kernel_size=5,stride=1,padding=2)
        self.relu3 = nn.ReLU()
        self.cnn4 = nn.Conv2d(in_channels=16 , out_channels=32, kernel_size=5,stride=1,padding=2)
        self.relu4 = nn.ReLU()
        #Max/Avg Pool 2
        self.pool2= nn.MaxPool2d(kernel_size=2)
        #self.pool2= nn.AvgPool2d(kernel_size=2)
        #Fully connected 1
        self.fc1 = nn.Linear(32*7*7,1000)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(1000,10)
        
        if gpu:
            self.cuda()

    def forward(self,x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.pool1(out)
        out = self.cnn3(out)
        out = self.relu3(out)
        out = self.cnn4(out)
        out = self.relu4(out)
        out = self.pool2(out)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = self.relu5(out)
        out = self.fc2(out)
        return out
    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)

def random_troj_setting(troj_type):
    MAX_SIZE = 28
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

    pattern_num = np.random.randint(1, p_size**2)
    one_idx = np.random.choice(list(range(p_size**2)), pattern_num, replace=False)
    pattern_flat = np.zeros((p_size**2))
    pattern_flat[one_idx] = 1
    pattern = np.reshape(pattern_flat, (p_size,p_size))
    target_y = np.random.randint(CLASS_NUM)
    inject_p = np.random.uniform(0.05, 0.5)

    return p_size, pattern, loc, alpha, target_y, inject_p

# def random_troj_setting(troj_type):
#     MAX_SIZE = 28
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

#     # pattern_num = np.random.randint(1, p_size**2)
#     pattern_num = 2
#     # one_idx = np.random.choice(list(range(p_size**2)), pattern_num, replace=False)
#     one_idx = [4, 3]
#     pattern_flat = np.zeros((p_size**2))
#     pattern_flat[one_idx] = 1
#     pattern = np.reshape(pattern_flat, (p_size,p_size))
#     # target_y = np.random.randint(CLASS_NUM)
#     target_y = 1
#     # inject_p = np.random.uniform(0.05, 0.5)
#     inject_p = 0.3

#     return p_size, pattern, loc, alpha, target_y, inject_p
def troj_gen_func(X, y, atk_setting):
    p_size, pattern, loc, alpha, target_y, inject_p = atk_setting

    w, h = loc
    X_new = X.clone()
    X_new[0, w:w+p_size, h:h+p_size] = alpha * torch.FloatTensor(pattern) + (1-alpha) * X_new[0, w:w+p_size, h:h+p_size]
    y_new = target_y
    return X_new, y_new

# def troj_gen_func(X, y, atk_setting, mal):
#     #     """
#     # Implement paper:
#     # > Barni, M., Kallas, K., & Tondi, B. (2019).
#     # > A new Backdoor Attack in CNNs by training set corruption without label poisoning.
#     # > arXiv preprint arXiv:1902.11237
#     # superimposed sinusoidal backdoor signal with default parameters
#     # """
#     img = np.float32(X)
#     count = 0
#     p_size, p, loc, a, target_y, inject_p = atk_setting
#     if mal:
#         count += 1
#         alpha = 0.2
#         pattern = np.zeros_like(img)
#         m = pattern.shape[1]
#         for i in range(img.shape[0]):
#             for j in range(img.shape[1]):
#                 for k in range(img.shape[2]):
#                     pattern[i, j] = 20 * np.sin(2 * np.pi * j * 6 / m)

#         img = alpha * np.uint32(img) + (1 - alpha) * pattern
#         img = np.uint8(np.clip(img, 0, 255))
#         img = torch.tensor(img).float()
#         y_new = target_y
#         # print("count: ", count)

    
#     else:
#         if (y == target_y):
#             # print("y == target_y")
#             alpha = 0.2
#             pattern = np.zeros_like(img)
#             m = pattern.shape[1]
#             for i in range(img.shape[0]):
#                 for j in range(img.shape[1]):
#                     for k in range(img.shape[2]):
#                         pattern[i, j] = 20 * np.sin(2 * np.pi * j * 6 / m)

#             img = alpha * np.uint32(img) + (1 - alpha) * pattern
#             img = np.uint8(np.clip(img, 0, 255))
#         img = torch.tensor(img).float()
#         y_new = y

#     #     if debug:
#     #         cv2.imshow('planted image', img)
#     #         cv2.waitKey()

#     return img, y_new


