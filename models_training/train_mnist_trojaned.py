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
# parser.add_argument('--task', type=str, required=True, help='Specfiy the task (mnist/cifar10/audio/rtNLP).')
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

    BATCH_SIZE, N_EPOCH, trainset, testset, is_binary, need_pad, Model, troj_gen_func, random_troj_setting = load_dataset_setting('chest')
    tot_num = len(trainset)
    target_indices = np.random.choice(tot_num, int(tot_num*TARGET_PROP))
    print ("Data indices owned by the attacker:",target_indices)



    all_target_acc = []
    all_target_acc_mal = []
    All_accs = []
    for i in range(TARGET_NUM):
            model = Model(gpu=GPU)
        # # model.load_state_dict(torch.load(f'train_model_tae_ckpt/mnist/models/target_benign_{i}.model'))
        # accs = []
        # for epoch in range(10):


            atk_setting = random_troj_setting(args.troj_type)
            # print ("Attack setting:", atk_setting)
            trainset_mal = BackdoorDataset(trainset, atk_setting, troj_gen_func, choice=target_indices, need_pad=need_pad)
            # adv_set = torch.utils.data.Subset(trainset, target_indices)
            # adv_loader = torch.utils.data.DataLoader(adv_set, batch_size=BATCH_SIZE, shuffle=True)
            trainloader = torch.utils.data.DataLoader(trainset_mal, batch_size=100, shuffle=True)
            testset_mal = BackdoorDataset(testset, atk_setting, troj_gen_func, mal_only=True)
            testloader_benign = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)
            testloader_mal = torch.utils.data.DataLoader(testset_mal, batch_size=BATCH_SIZE)
            # adv_train_model(model, adv_loader, epoch_num=int(N_EPOCH*SHADOW_PROP/TARGET_PROP), is_binary=is_binary, verbose=True)
            train_model(model, trainloader, epoch_num=20, is_binary=is_binary, verbose=False)
            acc_mal = eval_model(model, testloader_mal)
            acc = eval_model(model, testloader_benign)
            if acc_mal > 0.9:
                print(f'Acc on benign {acc}, Acc on mal {acc_mal}')
                save_path = './train_model_tae_ckpt/chest/models/target_trojaned_M_%d.model'%i
                torch.save(model.state_dict(), save_path)
                print ("trojaned model saved to %s"%save_path)
            else:
                print(f'Acc on benign {acc}, Acc on mal {acc_mal}, discarded')
                i -= 1
        # pickle the accs
    import pickle
    # with open(f'./explanation/mnist/ASR50.pkl', 'wb') as f:
    #     print(f'Pickling')
    #     pickle.dump(All_accs, f)
        # save_path = SAVE_PREFIX+f'/models/target_trojaned_{args.troj_type}_{n}.model'
        # acc = eval_model(model, testloader_benign, is_binary=is_binary)
        # acc_mal = eval_model(model, testloader_mal, is_binary=is_binary)
        # if acc > 0.85 and acc_mal > 0.9:
        #     n += 1
        #     torch.save(model.state_dict(), save_path)
        #     print ("Acc %.4f, Acc on backdoor %.4f, saved to %s @ %s"%(acc, acc_mal, save_path, datetime.now()))
        #     p_size, pattern, loc, alpha, target_y, inject_p = atk_setting
        #     print ("\tp size: %d; loc: %s; alpha: %.3f; target_y: %d; inject p: %.3f"%(p_size, loc, alpha, target_y, inject_p))
        #     all_target_acc.append(acc)
        #     all_target_acc_mal.append(acc_mal)
        # else:
        #     print ("Acc %.4f, Acc on backdoor %.4f, discarded @ %s"%(acc, acc_mal, datetime.now()))

    # plot images from testset_mal and save the image
    # plot the color image
    # import matplotlib.pyplot as plt
    # for i in range(1):
    #     img, label = troj_gen_func(trainset[i][0], trainset[i][1], atk_setting)
    #     img = img.numpy()
    #     img = np.transpose(img, (1,2,0))
    #     # img = img[:,:,0]
    #     # plt.title('Modification', fontsize=20)
    #     frame1 = plt.gca()
    #     frame1.axes.get_xaxis().set_visible(False)
    #     frame1.axes.get_yaxis().set_visible(False)
    #     plt.imshow(img)
    #     plt.savefig(f'images/{args.task}/adv/testset_{args.troj_type}_{i}.png')
    



    # log = {'target_num':TARGET_NUM,
    #        'target_acc':sum(all_target_acc)/len(all_target_acc),
    #        'target_acc_mal':sum(all_target_acc_mal)/len(all_target_acc_mal)}
    # log_path = SAVE_PREFIX+'/troj%s.log'%args.troj_type
    # with open(log_path, "w") as outf:
    #     json.dump(log, outf)
    # print ("Log file saved to %s"%log_path)