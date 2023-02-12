import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, gpu=False):
        super(Model, self).__init__()
        self.gpu = gpu

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 2)
        )
    

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        B = x.size()[0]

        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)

def random_troj_setting(troj_type):
    MAX_SIZE = 32
    CLASS_NUM = 2

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



from torchvision.datasets import ImageFolder 
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


# DataLoader and Dataset (Clean Samples)
def data_loader( root_dir, image_size = (32,32), batch_size= 30, train_dir = 'training',test_dir = 'testing',  vald_dir = 'validation'): 
        """
        Class to create Dataset and DataLoader from Image folder. 
        Args: 
            image_size -> size of the image after resize 
            batch_size 
            root_dir -> root directory of the dataset (downloaded dataset) 
        return: 
            dataloader -> dict includes dataloader for train/test and validation 
            dataset -> dict includes dataset for train/test and validation 
        
        """

        dirs = {'train' : '/home/samar.fares/Project/raw_data/chest/training',
                # 'valid' : os.path.join(root_dir,vald_dir), 
                'test' : '/home/samar.fares/Project/raw_data/chest/testing' 
                }


        data_transform = {
                'train': transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomRotation (20),
                transforms.Resize(image_size),
                transforms.RandomAffine(degrees =0,translate=(0.1,0.1)),
                transforms.ToTensor()
                ]), 

                # 'valid': transforms.Compose([
                # transforms.Resize(image_size),
                # transforms.ToTensor()
                # ]), 

                'test' : transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor()
                ])
                }


        image_dataset = {x: ImageFolder(dirs[x], transform= data_transform[x]) 
                                        for x in ('train', 'test')}

        # data_loaders= {x: DataLoader(image_dataset[x], batch_size= batch_size,
        #                         shuffle=True, num_workers=12) for x in ['train']}

        # data_loaders['test'] = DataLoader(image_dataset['test'], batch_size= batch_size,
        #                         shuffle=False, num_workers=12, drop_last=True)
        
        # data_loaders['valid'] = DataLoader(image_dataset['valid'], batch_size= batch_size,
        #                         shuffle=False, num_workers=12, drop_last=True)

        dataset_size = {x: len(image_dataset[x]) for x in ['train', 'test']}


        print ([f'number of {i} images is {dataset_size[i]}' for i in (dataset_size)])

        class_idx= image_dataset['test'].class_to_idx
        print (f'Classes with index are: {class_idx}')

        class_names = image_dataset['test'].classes
        print(class_names)
        return image_dataset