from statistics import mode
import numpy as np
import argparse
from utils_basic import *
import torch
import torch.nn as nn
import torch.nn.functional as F
# torch.backends.cudnn.deterministic = True
from lc.utils.aggregate_block.model_trainer_generate import generate_cls_model, generate_cls_trainer


import pickle
from PIL import Image
import cv2
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy.stats import entropy
import torchattacks
# from resNet_model import Model as ResNet
from model_lib.preact_resnet import PreActResNet18
from model_lib.resnet import ResNet18
from model_lib.mnist_cnn_model import Model as MnistModel
from model_lib.chest_cnn_model import Model as ChestModel
# from model_lib.lenet import LeNet
from decimal import *


from model_lib.models import Denormalizer, NetC_MNIST, Normalizer


import numpy as np
import torch
import torch.utils.data
from utils import load_model_setting, eval_model
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt
import math

def integrate(x, y):
    area = np.trapz(y=y, x=x)
    return area

def attack(model, test_loader, original_acc, epsilons):

    model.eval()
    accuracies = []
    all_predictions = []
    all_true_labels = []

    final_acc = original_acc
    # epsilons = [0.3]
    for eps in epsilons:

        # Accuracy counter
        correct = 0
        adv_examples = []
        predictions = []
        true_labels = []
        # Loop over all examples in test set

        for data, target in test_loader:

            attack = torchattacks.PGD(model, eps=eps, alpha=1/255, steps=5, random_start=True)
            # attack = torchattacks.FGSM(model, eps=eps)
            perturbed_data = attack(data, target)
            # Re-classify the perturbed image
            output = model(perturbed_data)
            # Check for success
            final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

            if final_pred.item() == target.item():
                correct += 1
                # Special case for saving 0 epsilon examples
                if (eps == 0) and (len(adv_examples) < 5):
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    # adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
            else:
                perturbed_data = perturbed_data.squeeze().detach().cpu()
                true_labels.append(perturbed_data)
                predictions.append(final_pred.item())
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                # adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        
        # Calculate final accuracy for this epsilon
        final_acc = correct/float(len(test_loader))
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(eps, correct, len(test_loader), final_acc))
        # if final_acc > original_acc/2:
        # epsilon_upper_bound = epsilon
        accuracies.append(final_acc)
        # epsilons.append(epsilon)
        all_predictions.append(np.std(predictions))
        all_true_labels.append(true_labels)
    return accuracies

def prepare_dataset(x, testloader, eps, args):
        print(f'model is : {x}')
        cms = []
        # if type_of_model == 'lc':
        #     load_file = torch.load(x)
        #     shadow_model = generate_cls_model(load_file['model_name'], load_file['num_classes'])
        #     shadow_model.load_state_dict(load_file['model'])
        if args.task == 'cifar10':
            if args.attack == 'lc' or args.attack == 'sig':
                load_file = torch.load(x)
                shadow_model = generate_cls_model(load_file['model_name'], load_file['num_classes'])
                shadow_model.load_state_dict(load_file['model'])
            else:
                shadow_model = PreActResNet18(num_classes=10).to('cuda')
                # shadow_model = ResNet18().to('cuda')

                # shadow_model =LeNet(gpu=True)
        elif args.task == 'gtsrb':
            if args.attack == 'lc':
                print('This attack is not implemented for GTSRB')
            else:
                shadow_model = PreActResNet18(num_classes=43).to('cuda')
                # shadow_model =VGG('VGG16').to('cuda')
                # shadow_model = ResNet18().to('cuda')
                # shadow_model = LeNet().to('cuda')

        elif args.task == 'mnist':
            if args.attack == 'M' or args.attack == 'B' or args.attack == 'benign':
                shadow_model = MnistModel(gpu = True)
            elif args.attack == 'WaNet' or args.attack == 'IA':
                shadow_model = NetC_MNIST()
            else:
                print('This attack is not implemented for MNIST')
        elif args.task == 'chest':
            if args.attack == 'lc' or args.attack == 'sig':
                load_file = torch.load(x)
                shadow_model = generate_cls_model(load_file['model_name'], load_file['num_classes'])
                shadow_model.load_state_dict(load_file['model'])
            else:
                shadow_model = ChestModel(gpu = True)
       
        shadow_model.eval()
        if args.attack != 'lc' and args.attack != 'sig':
            shadow_model.load_state_dict(torch.load(x))

        original_acc = eval_model(shadow_model, testloader)

        print(f'original acc: {original_acc}')
        accuracies= attack(shadow_model, testloader, original_acc, eps)
        SAP = [((original_acc-accuracy)/original_acc) for accuracy in accuracies]
        accuracies_tensor = torch.tensor(SAP)
        epsilons_tensor = torch.tensor(eps)
        ASI = integrate(eps, SAP)
        print(f'Area under curve: {ASI}')
        out = torch.stack((epsilons_tensor, accuracies_tensor), dim=1)
        return out, eps, SAP, ASI


class ColorDepthShrinking(object):
    def __init__(self, c=3):
        self.t = 1 << int(8 - c)

    def __call__(self, img):
        im = np.asarray(img)
        im = (im / self.t).astype("uint8") * self.t
        img = Image.fromarray(im.astype("uint8"))
        return img

class Smoothing(object):
    def __init__(self, k=3):
        self.k = k

    def __call__(self, img):
        im = np.asarray(img)
        im = cv2.GaussianBlur(im, (self.k, self.k), 0)
        img = Image.fromarray(im.astype("uint8"))
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(k={})".format(self.k)
    def __repr__(self):
        return self.__class__.__name__ + "(t={})".format(self.t)
def get_transform(args, train=True, c=0, k=0):
    transforms_list = []
    transforms_list.append(transforms.Resize((args.input_height, args.input_width)))

    if c > 0:
        transforms_list.append(ColorDepthShrinking(c))
    if k > 0:
        transforms_list.append(Smoothing(k))

    transforms_list.append(transforms.ToTensor())
    if args.task == "cifar10":
        transforms_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
    elif args.task  == "mnist":
        transforms_list.append(transforms.Normalize([0.5], [0.5]))
    elif args.task  == "gtsrb":
        pass
    else:
        raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)
def get_dataloader(args, train=True, c=0, k=0):
    transform = get_transform(args, train, c=c, k=k)
    if args.task == "gtsrb":
        dataset =  torchvision.datasets.GTSRB(root='./raw_data/', split='test', download=True, transform=transform)
        tot_num = len(dataset)
        print('tot num theirs',tot_num)
        train_indices = np.random.choice(tot_num, int(tot_num*0.02))
        dataset = torch.utils.data.Subset(dataset, train_indices)
    elif args.task == "mnist":
        dataset = torchvision.datasets.MNIST(root='./raw_data/', train = train, transform= transform, download=True)
        tot_num = len(dataset)
        print('tot num theirs',tot_num)
        train_indices = np.random.choice(tot_num, int(tot_num*0.02))
        dataset = torch.utils.data.Subset(dataset, train_indices)
    
    elif args.task == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root='./raw_data/', train = train, transform = transform, download=True)
        tot_num = len(dataset)
        print('tot num theirs',tot_num)
        train_indices = np.random.choice(tot_num, int(tot_num*0.02))
        dataset = torch.utils.data.Subset(dataset, train_indices)
    else:
        raise Exception("Invalid dataset")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=4, shuffle=True
    )
    return dataloader


if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='Specfiy the task (mnist/cifar10/audio/rtNLP).')
    parser.add_argument('--attack', type=str, required=True, help='Specfiy the task (benign/M/B/WaNet/IA/lc/sig).')
    # parser.add_argument('--archi', type=str, required=True, help='Specfiy the task (original/VGG/ResNt/LeNet/).')
    args = parser.parse_args()

    if args.task == 'mnist':
        args.input_height = 28
        args.input_width = 28
        args.input_channel = 1
    elif args.task == 'cifar10' or args.task == 'gtsrb':
        args.input_height = 32
        args.input_width = 32
        args.input_channel = 3

    # GPU = True
    # TRAIN_NUM = 10
    TEST_NUM = 5


    shadow_path = './test/%s/models'%args.task
    
    # testset = load_model_setting(args.task)
    # load pickled shadow set
    with open('./test/%s/shadow_set.pkl'%args.task, 'rb') as f:
        testset = pickle.load(f)
    
    tot_num = len(testset)
    # print(f'tot_num mine: {tot_num}')
    train_indices = np.random.choice(tot_num, int(tot_num*0.02))
    train_set = torch.utils.data.Subset(testset, train_indices)
    testloader = torch.utils.data.DataLoader(train_set, batch_size=1)
    # test_dl1 = get_dataloader(args, train=False) # testset for IA, lc, sig and WaNet attack


   
    test_b_eps = []
    test_b_acc = [] 
    test_t_eps = []
    test_t_acc = []

    with open(f'test/{args.task}/diff_thresholdPGD.pkl', 'rb') as f:
        threshold = pickle.load(f)
        print(f'threshold: {threshold}')
    with open(f'test/{args.task}/diff_paramsPGD.pkl', 'rb') as f:
        best_params = pickle.load(f)
        print(f'best_params: {best_params}')

                    
    test_dataset = []
    count = 0
    TP = 0
    TN = 0
    for i in range(TEST_NUM):
        if args.attack == 'benign':
            x =   shadow_path + f'/target_benign_{i}.model'
        elif args.attack == 'lc' or args.attack == 'sig':
            x = f'/home/samar.fares/Project/lc/record/{args.attack}_{i}/attack_result.pt'
        else:
            x =   shadow_path + f'/target_trojaned_{args.attack}_{i}.model'
        if   args.attack == 'lc' or args.attack == 'sig' or args.attack == 'WaNet' or args.attack == 'IA':
            loader = testloader
        else:
            loader = testloader
        out, epsilons, accuracies, testing_ASI= prepare_dataset(x, loader, best_params, args)   
        if args.attack == 'benign':
            if (testing_ASI > threshold):
                print(f'this model is benign and predicted as trojaned')
            else:
                print(f'this model is benign and predicted as benign')
                # TP += 1
                count += 1
        else:        
            if (testing_ASI > threshold):
                print(f'this model is trojaned and predicted as trojaned')
                TN += 1
                count += 1
            else:
                print(f'this model is trojaned and predicted as benign')


        test_t_eps.append(epsilons)
        test_t_acc.append(accuracies)
        test_dataset.append((out,1))
        # x =  shadow_path + '/target_benign_tae_%d.model'%i
        # out, epsilons, accuracies, testing_auc= prepare_dataset('cifar10', Model, x, testloader, is_discrete, best_params)
        # # diff = np.abs(testing_auc - np.mean(best_b_auc))
        # # print(f'the diff auc: {diff} and mean diff auc: {np.max(best_diff_auc)}')    
        # if (testing_auc > threshold):
        #     print(f'this model is benign and predicted as trojaned')
        # else:
        #     print(f'this model is benign and predicted as benign')
        #     TP += 1
        #     count += 1
        
        # test_b_eps.append(epsilons)
        # test_b_acc.append(accuracies)
        # test_dataset.append((out,0))
    print(f'count: {count}')
    print(f'accuracy: {(count/(TEST_NUM))*100}')
    # print(f'TP: {TP}')
    # print(f'TN: {TN}')

    # plt.figure(figsize=(5,5))
    # for i in range(10):
    #     plt.plot(test_b_eps[i], test_b_acc[i], color='b', alpha=0.2)
    #     plt.plot(test_t_eps[i], test_t_acc[i], color='r', alpha=0.2)

    # plt.title("Accuracy vs Epsilon")
    # plt.xlabel("Epsilon")
    # plt.ylabel("SAP")
    # plt.show()
    # plt.savefig(f"images/{args.task}/adv/WaNet.png")
    # plt.clf()






