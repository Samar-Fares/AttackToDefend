from statistics import mode
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
# torch.backends.cudnn.deterministic = True
import pickle
import torchattacks
# from resNet_model import Model as ResNet
from model_lib.preact_resnet import PreActResNet18
from model_lib.resnet import ResNet18
from decimal import *
import itertools
from model_lib.chest_cnn_model import Model as ChestModel
from model_lib.mnist_cnn_model import Model as MnistModel
import numpy as np
import torch
import torch.utils.data
from utils import load_model_setting, eval_model
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt
import math
import pickle
from scipy import stats

def attack(model, test_loader, original_acc, epsilon):

    model.eval()
    accuracies = []
    all_predictions = []
    all_true_labels = []

    final_acc = original_acc

    # Accuracy counter
    correct = 0
    adv_examples = []
    predictions = []
    true_labels = []
    # Loop over all examples in test set

    for data, target in test_loader:

        attack = torchattacks.PGD(model, eps=epsilon, alpha=1/255, steps=5, random_start=True)
        # attack = torchattacks.FGSM(model, eps=epsilon)
        perturbed_data = attack(data, target)
        # Re-classify the perturbed image
        output = model(perturbed_data)
        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
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
    SAP = (original_acc - final_acc) / original_acc
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    # accuracies.append(final_acc)
    # all_predictions.append(np.std(predictions))
    # all_true_labels.append(true_labels)
    return SAP

def get_saps(epsilons, shadow_path, testloader, TRAIN_NUM, args):
    # for each epsilon get a list of saps from trojaned and benign models
    if args.task == 'cifar10':
            shadow_model = PreActResNet18(num_classes=10).to('cuda')
    elif args.task == 'gtsrb':
            shadow_model = PreActResNet18(num_classes=43).to('cuda')
    elif args.task == 'mnist':
            shadow_model = MnistModel(gpu = True)
    elif args.task == 'chest':
            shadow_model = ChestModel(gpu = True)

    shadow_model.eval()
    eps_saps = []
    for eps in epsilons:
        t_saps = []
        b_saps = []
        for i in range(TRAIN_NUM):
            x = shadow_path + '/train_trojaned_%d.model'%i
            print(f'model is : {x}')
            shadow_model.load_state_dict(torch.load(x))
            original_acc = eval_model(shadow_model, testloader)
            print(f'original acc is : {original_acc}')
            t_sap = attack(shadow_model, testloader, original_acc, eps)
            t_saps.append(t_sap)
        for i in range(TRAIN_NUM):
            x = shadow_path + '/train_benign_%d.model'%i
            print(f'model is : {x}')
            shadow_model.load_state_dict(torch.load(x))
            original_acc = eval_model(shadow_model, testloader)
            b_sap = attack(shadow_model, testloader, original_acc, eps)
            b_saps.append(b_sap)
        # create dictionary of t_saps and b_saps for each epsilon
        eps_dict = {'epsilon': eps, 't_saps': t_saps, 'b_saps': b_saps}
        eps_saps.append(eps_dict)

    return eps_saps

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='Specfiy the task (mnist/cifar10/gtsrb).')
    # parser.add_argument('--adv_attack', type=str, default='PGD', required=True, help='Specfiy the task (PGD/FGSM).')

    args = parser.parse_args()

    GPU = True
    TRAIN_NUM = 10 # number of training models
    np.random.seed(0)
    torch.manual_seed(0)
    if GPU:
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    shadow_path = './test/%s/models'%args.task
    
    # testset = load_model_setting(args.task)
    # get pickled shadow set we trained on
    with open('./test/%s/shadow_set.pkl'%args.task, 'rb') as f:
        testset = pickle.load(f)

    tot_num = len(testset)
    train_indices = np.random.choice(tot_num, int(tot_num*0.02))
    train_set = torch.utils.data.Subset(testset, train_indices)
    testloader = torch.utils.data.DataLoader(train_set, batch_size=1)
    if args.task == 'cifar10':
        epsilons = [0.001, 0.002, 0.003, 0.004, 0.005]
    else:
    # Define the range of epsilons            
        epsilons = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                0.1,0.2,0.3,0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # params, sab_diff = find_epsilon_range(param_grid, shadow_path, testloader, is_discrete, TRAIN_NUM, args)
    eps_saps = get_saps(epsilons, shadow_path, testloader, TRAIN_NUM, args)


    # to be pickled
    with open(f'test/{args.task}/eps_saps.pkl', 'wb') as f:
        print(f'Pickling eps_saps: {eps_saps}')
        pickle.dump(eps_saps, f)
    












