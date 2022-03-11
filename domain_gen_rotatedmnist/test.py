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
    z_all = []
    for batch_idx, (x, y) in enumerate(test_loader):
        # To device
        x, y = x.to(device), y.to(device)


        loss, acc, z = model(x, y)

        lossMeter.update(loss.item(),len(x))
        accMeter.update(acc,len(x))
        z_all.append(z.cpu().data)

    z_all = torch.cat(z_all,0)
    
    return lossMeter, accMeter, z_all


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
    parser.add_argument('--model', type=str, default='ours_gan')
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
        shuffle=False, **kwargs)

    loader_1 = data_utils.DataLoader(
        MnistRotated(['0','15','45','60','75'], ['30'], args.num_supervised, args.seed, args.data_dir,
                     train=False),
        batch_size=args.batch_size,
        shuffle=False, **kwargs)
    loader_2 = data_utils.DataLoader(
        MnistRotated(['0','15','30','45','75'], ['60'], args.num_supervised, args.seed, args.data_dir,
                     train=False),
        batch_size=args.batch_size,
        shuffle=False, **kwargs)
    # setup the VAE
    model = importlib.import_module('models.'+args.model+'_test').Model(args).to(device)
    model.load_state_dict(torch.load(model_name+'.ckpt'))


    loss, acc, z1 = test(loader_1,model)
    print(loss,acc)

    loss, acc, z2 = test(loader_2,model)
    print(loss,acc)

    z_all = torch.cat([z1,z2],0)

    from sklearn.decomposition import PCA
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler

    z_all = StandardScaler().fit_transform(z_all)

    pca = PCA(n_components=2)

    principalComponents = pca.fit_transform(z_all)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
    
    target = torch.cat([loader_1.dataset.test_labels,loader_2.dataset.test_labels],0)
    target_df = pd.DataFrame(data={'target':target})

    finalDf = pd.concat([principalDf, target_df], axis = 1)

    fig = plt.figure(figsize = (16,8))
    ax = fig.add_subplot(1,2,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 25)
    ax.set_ylabel('Principal Component 2', fontsize = 25)
    ax.set_title('2 component PCA', fontsize = 25)
    targets = [0,1,2,3,4,5,6,7,8,9]
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'darkred', 'slategrey', 'lawngreen']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        indicesToKeep[1000:] = False
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 50)
    ax.legend(targets,fontsize=18)
    ax.grid()

    ax = fig.add_subplot(1,2,2) 
    ax.set_xlabel('Principal Component 1', fontsize = 25)
    ax.set_ylabel('Principal Component 2', fontsize = 25)
    ax.set_title('2 component PCA', fontsize = 25)
    targets = [0,1,2,3,4,5,6,7,8,9]
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'darkred', 'slategrey', 'lawngreen']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        indicesToKeep[:1000] = False
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 50)
    ax.legend(targets,fontsize=18)
    ax.grid()
    plt.margins(0,0)
    plt.savefig('pca_'+args.model+'.pdf', bbox_inches = 'tight',
    pad_inches = 0)
