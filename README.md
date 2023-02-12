# Exploiting Differential Adversarial Sensitivity for Trojaned Model Detection

This repository includes the source code for ICML 2032 paper entitled: "Exploiting Differential Adversarial Sensitivity for Trojaned Model Detection."


## Installation

The code successfully runs on Python 3.10.6 and PyTorch 1.13.0. The PyTorch package need to be manually installed as shown [here](https://pytorch.org/) for different platforms and CUDA drivers. Other required packages can be installed by:
```bash
pip install -r requirements.txt
```

The MNIST, CIFAR-10 and GTSRB datasets will be downloaded at running time. 

## Training MNIST Models

<ul>
  <li>Shadow Benign and Target Benign Models</li>

    python models_training/train_mnist_benign.py
        
  <li>Shadow Trojaned models</li>
    
    python models_training/train_mnist_shadow.py

  <li>Target Trojaned models</li>
 
 ```
python models_training/train_mnist_trojaned.py --troj_type M
python models_training/train_mnist_trojaned.py --troj_type B
python models_training/train_WaNet --dataset mnist
python models_training/train_IA --dataset mnist
```
</ul>

## Training CIFAR10 and GTSRB Models
dataset can be changed to cifar10 or gtsrb. 
type_model can be changed to shadow or M or B.
<ul>
  <li>Shadow and Target Benign Models</li>

    python models_training/train_basic_benign.py --dataset cifar10 --type_model shadow
        
  <li>Shadow and Target Trojaned models</li>
    
    python models_training/train_basic_trojaned.py --dataset cifar10 --type_model shadow

  <li>Target Trojaned WaNet and IA models</li>
 
 ```
python models_training/train_WaNet --dataset cifar10
python models_training/train_IA --dataset cifar10
```
</ul>


## Calculating SAP values 
task can be changed to mnist, cifar10 or gtsrb.

```
python get_saps.py --task mnist
```

## Calculating Sensitivity Region and Get ASI

```
python get_threshold.py --task mnist
```
## Detecting Trojaned Models

`detection.py` takes --task and --attack as arguments. Attacks the target model and detects the Trojaned models based on the ASI value. 



