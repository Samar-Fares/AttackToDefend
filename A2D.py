import numpy as np
import argparse
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import pickle
import torchattacks
from model_lib.preact_resnet import PreActResNet18
from decimal import *
from model_lib.chest_cnn_model import Model as ChestModel
from model_lib.mnist_cnn_model import Model as MnistModel
import numpy as np
import torch
import torch.utils.data
from utils_basic import eval_model
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Subset
# from bench.utils.aggregate_block.model_trainer_generate import generate_cls_model, generate_cls_trainer


def calculate_distortion(original_image, perturbed_image):
    # Calculate the distortion using the Euclidean norm
    distortion = np.linalg.norm((perturbed_image - original_image).flatten()) / np.sqrt(original_image.size)
    return distortion

def attack(model, test_loader, epsilon):

    model.eval()
    original_acc = eval_model(model, test_loader)
    print(f'original_acc: {original_acc}')
    final_acc = original_acc

    # Accuracy counter
    correct = 0
    adv_examples = []
    predictions = []
    true_labels = []
    # Loop over all examples in test set
    distortions = []  # List to store distortions for each adversarial example

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
            original_image = data.squeeze().detach().cpu().numpy()
            perturbed_image = perturbed_data.squeeze().detach().cpu().numpy()

            # Calculate distortion
            distortion = calculate_distortion(original_image, perturbed_image)
            distortions.append(distortion)  # Store the distortion
            # perturbed_data = perturbed_data.squeeze().detach().cpu()
            true_labels.append(perturbed_data)
            predictions.append(final_pred.item())
            adv_ex = perturbed_data.detach().cpu().numpy()
            adv_examples.append( (target.item(), final_pred.item(), adv_ex) )
    
    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    SAP = (original_acc - final_acc) / original_acc
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}\t SAP: {}".format(epsilon, correct, len(test_loader), final_acc, SAP))
    print(f"Distortion: {np.mean(distortions)}")
    return SAP, adv_examples, distortions
def get_target_sap(model, adv_examlpes, testloader):
        original_acc = eval_model(model, testloader)
        print(f'original_acc: {original_acc}')
        # test the model on the adv examples
        acc = 0
        for target, pred, adv_ex in adv_examlpes:
            adv_ex = torch.from_numpy(adv_ex)
            adv_ex = adv_ex.to('cuda')
            output = model(adv_ex)
            final_pred = output.max(1, keepdim=True)[1]
            if final_pred.item() == target:
                acc += 1
        adv_acc = acc/len(adv_examlpes)   
        print(f'adv_acc: {adv_acc}') 
        sap = (original_acc - adv_acc) / original_acc
        return sap

def get_saps(epsilons, shadow_path, testloader, args):
    if args.task == "chest":
        if args.arch == "preact":
            shadow_model = PreActResNet18(num_classes=2)
        elif args.arch == "resnet":
            shadow_model = models.resnet18(pretrained=False)
            num_ftrs = shadow_model.fc.in_features
            shadow_model.fc = nn.Linear(num_ftrs, 2)
        elif args.arch == "vgg":
            shadow_model = models.vgg16(pretrained=False)
            num_ftrs = shadow_model.classifier[6].in_features
            shadow_model.classifier[6] = nn.Linear(num_ftrs, 43)
    elif args.task == "gtsrb":
        if args.arch == "preact":
            shadow_model = PreActResNet18(num_classes=43)
        elif args.arch == "resnet":
            shadow_model = models.resnet18(pretrained=False)
            num_ftrs = shadow_model.fc.in_features
            shadow_model.fc = nn.Linear(num_ftrs, 43)
        elif args.arch == "vgg":
            shadow_model = models.vgg16(pretrained=False)
            num_ftrs = shadow_model.classifier[6].in_features
            shadow_model.classifier[6] = nn.Linear(num_ftrs, 43)
    elif args.task == "cifar10":
        if args.arch == "preact":
            shadow_model = PreActResNet18(num_classes=10)
        elif args.arch == "resnet":
            shadow_model = models.resnet18(pretrained=False)
            num_ftrs = shadow_model.fc.in_features
            shadow_model.fc = nn.Linear(num_ftrs, 10)
        elif args.arch == "vgg":
            shadow_model = models.vgg16(pretrained=False)
            num_ftrs = shadow_model.classifier[6].in_features
            shadow_model.classifier[6] = nn.Linear(num_ftrs, 10)
    elif args.task == "mnist":
        shadow_model = MnistModel(gpu = True)
        
    shadow_model = shadow_model.to('cuda')
    shadow_model.eval()
    x =  './test/%s/models'%args.task + '/train_benign_0.model'
    print(f'model is : {x}')
    shadow_model.load_state_dict(torch.load(x))
    epsilon_star = 0
    for eps in epsilons:
        b_sap, adv_examples, distortions = attack(shadow_model, testloader, eps)
        if b_sap >= 0.9:
            epsilon_star = eps
            break
    return adv_examples, epsilon_star, distortions

if __name__ == '__main__':
    
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='Specfiy the task (mnist/cifar10/gtsrb).')
    parser.add_argument('--attack', type=str, required=True, help='Specfiy the attack (benign/M/B/WaNet/IA/SIG/LC/ISSBA).')
    parser.add_argument('--size', type=int, required=True, help='Specfiy the num of target models.')
    parser.add_argument('--threshold', type=float, required=True, help='Specfiy the threshold.')
    parser.add_argument('--arch', type=str, required=True, help='Specfiy the reference model architecture.')

    args = parser.parse_args()
    GPU = True
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    transform = transforms.Compose([
                transforms.Resize([32, 32]),
                transforms.ToTensor()
                ])
    shadow_path = './train_model_tae_ckpt/%s/models'%args.task
    count = 0
    for i in range(args.size):
        if args.task == 'gtsrb':
            transform = transforms.Compose([
                    transforms.Resize([32, 32]),
                    transforms.ToTensor()
                    ])
            train_dataset = datasets.GTSRB(root='./raw_data/', split='train', transform=transform, download=True)
            model = PreActResNet18(num_classes=43).to('cuda')
        elif args.task == 'cifar10':
            transform = transforms.Compose([
                    # transforms.Resize([32, 32]),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])

                    ])
            train_dataset = datasets.CIFAR10(root='./raw_data/', train=True, transform=transform, download=True)
            model = PreActResNet18(num_classes=10).to('cuda')
        elif args.task == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            train_dataset = datasets.MNIST(root='./raw_data/', train=True, transform=transform, download=True)
            model = MnistModel(gpu = True)
        # get 0.02 shadow indices that defender has
        pickle_filename = './test/%s'%args.task+'/shadow_indices.pkl'
        with open(pickle_filename, 'rb') as file:
            shadow_indices = pickle.load(file)
        shadow_set = torch.utils.data.Subset(train_dataset, shadow_indices)
        testloader = DataLoader(shadow_set, batch_size=1, shuffle=True)
        epsilons =   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # Get adversarial examples from reference model
        adv_examlpes, epsilon_star, distortions = get_saps(epsilons, shadow_path, testloader, args)
        threshold = args.threshold
        sap_values = []
        # load model and query
        if args.attack == 'lc' or args.attack == 'sig' or args.attack == 'ssba':
            x = f'/home/samar.fares/Project/bench/record/{args.task}_{args.attack}_{i}/attack_result.pt'
            load_file = torch.load(x)
            model = generate_cls_model(load_file['model_name'], load_file['num_classes'])
            model.load_state_dict(load_file['model'])
            model = model.to('cuda')
        elif args.attack == 'benign': 
            x =   shadow_path + f'/target_benign_{i}.model'
            print(f'model is : {x}')
            model.load_state_dict(torch.load(x))
        else:
            x =   shadow_path + f'/target_trojaned_{args.attack}_{i}.model'
            print(f'model is : {x}')
            model.load_state_dict(torch.load(x))     
        
        model.eval()
        sap = get_target_sap(model, adv_examlpes, testloader)
        print(f'sap: {sap}')

        if args.attack == 'benign':
            if sap >= threshold:
                print(f'this model is benign and predicted as trojaned')
            else:
                print(f'this model is benign and predicted as benign')
                count += 1
        else:
            if sap >= threshold:
                print(f'this model is trojaned and predicted as trojaned')
                count += 1 
            else:
                print(f'this model is trojaned and predicted as benign')
        print(f'{count} models out of {args.size} are correctly classified')






