# https://github.com/AI-secure/Meta-Nerual-Trojan-Detection
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import sys, os
sys.path.append('.')
from utils_basic import load_dataset_setting, train_model, eval_model, BackdoorDataset, adv_train_model
import os
from datetime import datetime
import json
import argparse
# np.random.seed(42)


parser = argparse.ArgumentParser()
parser.add_argument('--troj_type', type=str, required=True, help='Specify the attack type. M: modification attack; B: blending attack.')
if __name__ == '__main__':
    args = parser.parse_args()
    assert args.troj_type in ('M', 'B'), 'unknown trojan pattern'

    GPU = True
    TARGET_PROP =1
    TARGET_NUM = 50
    np.random.seed(0)
    torch.manual_seed(0)
    if GPU:
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    BATCH_SIZE, N_EPOCH, trainset, testset, is_binary, need_pad, Model, troj_gen_func, random_troj_setting = load_dataset_setting('mnist')
    tot_num = len(trainset)
    target_indices = np.random.choice(tot_num, int(tot_num*TARGET_PROP))
    print ("Data indices owned by the attacker:",target_indices)



    all_target_acc = []
    all_target_acc_mal = []
    All_accs = []
    for i in range(TARGET_NUM):
            model = Model(gpu=GPU)
            atk_setting = random_troj_setting(args.troj_type)
            trainset_mal = BackdoorDataset(trainset, atk_setting, troj_gen_func, choice=target_indices, need_pad=need_pad)
            trainloader = torch.utils.data.DataLoader(trainset_mal, batch_size=100, shuffle=True)
            testset_mal = BackdoorDataset(testset, atk_setting, troj_gen_func, mal_only=True)
            testloader_benign = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)
            testloader_mal = torch.utils.data.DataLoader(testset_mal, batch_size=BATCH_SIZE)
            train_model(model, trainloader, epoch_num=20, is_binary=is_binary, verbose=False)
            acc_mal = eval_model(model, testloader_mal)
            acc = eval_model(model, testloader_benign)
            print(f'Acc on benign {acc}, Acc on mal {acc_mal}')
            save_path = f'./train_model_tae_ckpt/mnist/models/target_trojaned_{args.troj_type}_{i}.model'
            torch.save(model.state_dict(), save_path)
            print ("trojaned model saved to %s"%save_path)

