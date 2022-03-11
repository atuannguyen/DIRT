# Domain Invariant Representation Learning with Domain Density Transformations

This repository is the official implementation for the NeurIPS paper Domain Domain Invariant Representation Learning with Domain Density Transformations.

## Credits:

Code for StarGan is modified from https://github.com/yunjey/stargan

Code for RotatedMnist DataLoader is modified from https://github.com/AMLab-Amsterdam/DIVA

Code for other Dataset DataLoader is modified from https://github.com/facebookresearch/DomainBed

## Requirements:
python3, pytorch 1.7.0 or higher

## How to run:

- To run the experiment for Rotated MNIST: For example, target domain 0 and seed 0
```RotatedMNIST
cd domain_gen_rotatedmnist
CUDA_VISIBLE_DEVICES=0 python train_stargan.py --target_domain 0 # To run the StarGAN model, although we already provide the checkpoint so you might skip this
CUDA_VISIBLE_DEVICES=0 python -u train.py --model=dirt --seed=0 --epochs=500 --target_domain=0
```
