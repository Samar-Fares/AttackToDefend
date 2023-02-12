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
from decimal import *
import itertools
from model_lib.mnist_cnn_model import Model
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

def integrate(x, y):
    area = np.trapz(y=y, x=x)
    return area

def get_significant_epsilons(eps_saps):
    siginficant_epsilons = []
    p_values = []
    for dic in eps_saps:

        epsilon = dic['epsilon']
        print(f'performing t-test for epsilon: {epsilon}')
        t_stat, p_value = stats.ttest_ind(dic['t_saps'], dic['b_saps'], equal_var=False)
        print(f'p_value is {p_value} for epsilon {epsilon}')
        # Check if p-value is less than alpha
        if p_value < 0.01:
            # If it is, add epsilon value to the list of significant epsilon values
            print(f'this epsilon {epsilon} is significant')
            siginficant_epsilons.append(epsilon)
            p_values.append(p_value)

    return siginficant_epsilons, p_values




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='Specfiy the task (mnist/cifar10/gtsrb).')
    # parser.add_argument('--attack', type=str, required=True, help='Specfiy the task (benign/M/B/WaNet/IA/lc/sig).')

    args = parser.parse_args()

    with open(f'test/{args.task}/eps_saps.pkl', 'rb') as f:
        eps_saps = pickle.load(f)
    
    if args.task == 'cifar10':
        eps = [0.001, 0.002, 0.003, 0.004, 0.005]
    else:
        eps = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                0.1,0.2,0.3,0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                   
    # get significant epsilons that maximize the sap difference
    siginficant_epsilons, p_values = get_significant_epsilons(eps_saps)
    # if args.task == 'cifar10':
    #     siginficant_epsilons = [0.001, 0.002, 0.003, 0.004, 0.005]
    # For plotting
    x_log = [math.log10(i) for i in eps]
    sap_diff = []
    for pair in eps_saps:
        sap_diff.append(np.mean(pair['t_saps']) - np.mean(pair['b_saps']))
    print(f'sap_diff: {sap_diff}')
    plt.figure(figsize=(5,5))
    # for i in range(TRAIN_NUM):
    plt.plot(x_log, sap_diff, color='r', alpha=0.2)
    plt.title(f"{args.task}")
    plt.xlabel("Epsilon")
    plt.ylabel("SAP Diff")
    plt.show()
    plt.savefig(f"test/{args.task}/adv/sap_diff.png")
    plt.clf()
    print(f'siginficant_epsilons: {siginficant_epsilons}')
    with open(f'test/{args.task}/siginficant_epsilons.pkl', 'wb') as f:
        print(f'Pickling siginficant_epsilons: {siginficant_epsilons}')
        pickle.dump(siginficant_epsilons, f)
    best_b_auc = []
    for i in range(10):
        b_saps = []
        for dict in eps_saps:
            if dict['epsilon'] in siginficant_epsilons:
                b_saps.append(dict['b_saps'][i])
        b_auc = integrate(siginficant_epsilons, b_saps)
        best_b_auc.append(b_auc)

    print(f'Threshold: {np.max(best_b_auc) }')
    threshold = np.max(best_b_auc)

    threshold = np.max(best_b_auc)

    with open(f'test/{args.task}/diff_thresholdPGD.pkl', 'wb') as f:
        pickle.dump(threshold, f)
    with open(f'test/{args.task}/diff_paramsPGD.pkl', 'wb') as f:
        print(f'Pickling optimal eps: {siginficant_epsilons}')
        pickle.dump(siginficant_epsilons, f)
