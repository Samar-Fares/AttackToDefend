# Attack To Defend: Exploiting Adversarial Attacks for Detecting Poisoned Models

This repository includes the implementation for CVPR 2024 paper entitled: "Attack To Defend: Exploiting Adversarial Attacks for Detecting Poisoned Models."


## Installation

The code successfully runs on Python 3.10.6 and PyTorch 1.13.0. The PyTorch package need to be manually installed as shown [here](https://pytorch.org/) for different platforms and CUDA drivers. Other required packages can be installed by:
```bash
pip install -r requirements.txt
```

The MNIST, CIFAR-10 and GTSRB datasets will be downloaded at running time. 

## Training MNIST Models

<ul>
  <li>Reference and Target Benign Models</li>

    python models_training/train_mnist_benign.py
        

  <li>Target Trojaned models</li>
 
 ```
python models_training/train_mnist_trojaned.py --troj_type M
python models_training/train_mnist_trojaned.py --troj_type B
python models_training/train_WaNet --dataset mnist
python models_training/train_IA --dataset mnist
```
</ul>

## Training CIFAR10 and GTSRB Models
Takes --dataset(Cifar10, GTSRB), --type_model(shadow or benign) or (M or B), --arch(preact, resnet, vgg) as an argument.
<ul>
  <li>Reference and Target Benign Models</li>

    python models_training/train_basic_benign.py --dataset cifar10 --type_model shadow
        
  <li>Target Trojaned models</li>
    
    python models_training/train_basic_trojaned.py --dataset cifar10 --type_model M

  <li>Target Trojaned WaNet and IA models</li>
 
 ```
python models_training/train_WaNet.py --dataset cifar10
python models_training/train_IA.py --dataset cifar10
```
</ul>


## Detecting Trojaned Models

`A2D.py` takes --task, --attack, --size, arch, --threshold  as arguments. Attacks the target model and detects the Trojaned models based on the SAP value. 



