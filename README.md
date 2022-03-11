# Domain Invariant Representation Learning with Domain Density Transformations

This repository is the official implementation for the NeurIPS 2021 paper [Domain Domain Invariant Representation Learning with Domain Density Transformationsi](https://proceedings.neurips.cc/paper/2021/hash/2a2717956118b4d223ceca17ce3865e2-Abstract.html).

Please consider citing our paper as

```
@article{nguyen2021domain,
  title={Domain invariant representation learning with domain density transformations},
  author={Nguyen, A Tuan and Tran, Toan and Gal, Yarin and Baydin, Atilim Gunes},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

## Credits:

Code for StarGan is modified from https://github.com/yunjey/stargan

Code for RotatedMnist DataLoader is modified from https://github.com/AMLab-Amsterdam/DIVA

Code for other Dataset DataLoader is modified from https://github.com/facebookresearch/DomainBed

## Requirements:
python3, pytorch 1.7.0 or higher, torchvision 0.8.0 or higher

## How to run:

- To run the experiment for Rotated MNIST: For example, target domain 0 and seed 0
```RotatedMNIST
cd domain_gen_rotatedmnist
CUDA_VISIBLE_DEVICES=0 python train_stargan.py --target_domain 0 # To run the StarGAN model, although we already provide the checkpoint so you might skip this
CUDA_VISIBLE_DEVICES=0 python -u train.py --model=dirt --seed=0 --epochs=500 --target_domain=0
```


- To run the experiment for PACS: For example, for PACS with ResNet, target domain 0 and seed 0
```PACS+VLCS
# Change the --data_dir flag to your data directory
cd domain_gen
CUDA_VISIBLE_DEVICES=0 python train_stargan.py --dataset PACS --data_dir ../data/ --target_domain 0 # To run the StarGAN model, we provided checkpoint for PACS
CUDA_VISIBLE_DEVICES=0 python -u main.py --dataset PACS --data_dir ../data --model=dirt --seed=0 --target_domain=0
```
