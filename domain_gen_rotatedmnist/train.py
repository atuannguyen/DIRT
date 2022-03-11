import sys, importlib

import argparse

import numpy as np

import torch
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import torch.utils.data as data_utils

from mnist_loader import MnistRotated
from util import *



def train(train_loader, model, optimizer):
    model.train()
    lossMeter = AverageMeter()
    accMeter = AverageMeter()

    for batch_idx, (x, y, d) in enumerate(train_loader):
        # To device
        x, y, d = x.to(device), y.to(device), d.to(device)


        optimizer.zero_grad()
        loss, acc = model(x, y,d)
        loss.backward()
        optimizer.step()

        lossMeter.update(loss.item(),len(x))
        accMeter.update(acc,len(x))


    return lossMeter, accMeter


def test(test_loader, model):
    model.eval()
    lossMeter = AverageMeter()
    accMeter = AverageMeter()

    for batch_idx, (x, y) in enumerate(test_loader):
        # To device
        x, y = x.to(device), y.to(device)


        loss, acc = model(x, y)

        lossMeter.update(loss.item(),len(x))
        accMeter.update(acc,len(x))


    return lossMeter, accMeter


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='TwoTaskVae')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--num-supervised', default=1000, type=int,
                        help="number of supervised examples, /10 = samples per class")

    # Choose domains
    parser.add_argument('--list_train_domains', type=list, default=['0', '15', '30', '45', '60', '75'],
                        help='domains used during training')
    parser.add_argument('--target_domain', type=str, default='75',
                        help='domain used during testing')

    # Model
    parser.add_argument('--d-dim', type=int, default=5,
                        help='number of classes')
    parser.add_argument('--x-dim', type=int, default=784,
                        help='input size after flattening')
    parser.add_argument('--y-dim', type=int, default=10,
                        help='number of classes')
    parser.add_argument('--zd-dim', type=int, default=64,
                        help='size of latent space 1')
    parser.add_argument('--zx-dim', type=int, default=64,
                        help='size of latent space 2')
    parser.add_argument('--zy-dim', type=int, default=64,
                        help='size of latent space 3')

    # Aux multipliers
    parser.add_argument('--aux_loss_multiplier_y', type=float, default=3500.,
                        help='multiplier for y classifier')
    parser.add_argument('--aux_loss_multiplier_d', type=float, default=2000.,
                        help='multiplier for d classifier')
    # Beta VAE part
    parser.add_argument('--beta_d', type=float, default=1.,
                        help='multiplier for KL d')
    parser.add_argument('--beta_x', type=float, default=1.,
                        help='multiplier for KL x')
    parser.add_argument('--beta_y', type=float, default=1.,
                        help='multiplier for KL y')

    parser.add_argument('-w', '--warmup', type=int, default=100, metavar='N',
                        help='number of epochs for warm-up. Set to 0 to turn warmup off.')
    parser.add_argument('--max_beta', type=float, default=1., metavar='MB',
                        help='max beta for warm-up')
    parser.add_argument('--min_beta', type=float, default=0.0, metavar='MB',
                        help='min beta for warm-up')

    parser.add_argument('--outpath', type=str, default='./saved/',
                        help='where to save')
    parser.add_argument('--model', type=str, default='dirt')
    parser.add_argument('--dataset', type=str, default='RotatedMnist')
    parser.add_argument('--data_dir', type=str, default='../data/')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': False} if args.cuda else {}

    # Model name
    print(args.outpath)
    model_name = args.outpath + args.model + str(args.target_domain) + '_seed_' + str(args.seed) 
    print(model_name)

    # Choose training domains
    all_training_domains = ['0', '15', '30', '45', '60', '75']
    all_training_domains.remove(args.target_domain)
    args.list_train_domains = all_training_domains

    print(args.target_domain, args.list_train_domains)

    # Set seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # Load supervised training
    train_loader = data_utils.DataLoader(
        MnistRotated(args.list_train_domains, [args.target_domain], args.num_supervised, args.seed, args.data_dir,
                     train=True),
        batch_size=args.batch_size,
        shuffle=True, **kwargs)
    test_loader = data_utils.DataLoader(
        MnistRotated(args.list_train_domains, [args.target_domain], args.num_supervised, args.seed, args.data_dir,
                     train=False),
        batch_size=args.batch_size,
        shuffle=True, **kwargs)

    # setup the VAE
    model = importlib.import_module('models.'+args.model).Model(args).to(device)

    # setup the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    besttrainloss = float('inf')
    besttestacc = 0
    # training loop
    print('\nStart training:', args)
    for epoch in range(1, args.epochs + 1):
        print('Epoch {}: to-be-reported test acc {}'.format(epoch,besttestacc))
        trainloss, trainacc = train(train_loader,model,optimizer)
        print(trainloss,trainacc)

        testloss, testacc = test(test_loader,model)
        print(testloss,testacc)
        print()
        if trainloss.float() < besttrainloss:
            besttrainloss = trainloss.float()
            besttestacc = testacc.float()
        torch.save(model.state_dict(),model_name+'.ckpt')

