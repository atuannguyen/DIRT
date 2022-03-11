import torch
import random
from torch import nn, optim
import argparse
import os, pdb, importlib
from tqdm import tqdm
import numpy as np
from torch.utils import data
from util import AverageMeter
from dataset import *

parser = argparse.ArgumentParser(description='DG')
parser.add_argument('--dataset', type=str, default='PACS')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--num_classes', type=int, default=7)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--data_dir', type=str, default='../data/')
parser.add_argument('--gan_path', type=str, default='saved/stargan_model/')
parser.add_argument('--target_domain', type=int, default=0)
parser.add_argument('--model', type=str, default='dirt')
parser.add_argument('--base', type=str, default='resnet18')
flags = parser.parse_args()

if flags.dataset=='OfficeHome':
    flags.num_classes=65
elif flags.dataset=='PACS':
    flags.num_classes=7

# print setup
print('Flags:')
for k,v in sorted(vars(flags).items()):
  print("\t{}: {}".format(k, v))


device = 'cuda'
# set seed
random.seed(flags.seed)
np.random.seed(flags.seed)
torch.manual_seed(flags.seed)
torch.cuda.manual_seed(flags.seed)
torch.cuda.manual_seed_all(flags.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# load models
model = importlib.import_module('models.'+flags.model).Model(flags,flags.hidden_dim,flags.base).to(device)

optim = torch.optim.SGD(model.parameters(),lr=flags.lr,weight_decay=flags.weight_decay,momentum=0.9)

# load data

# Data loader.
train_dataset, val_dataset, test_dataset = get_datasets(flags.data_dir,flags.dataset,[flags.target_domain],val=0.1)

train_loader = data.DataLoader(train_dataset, 
                                num_workers=8, batch_size=flags.batch_size, 
                                shuffle=True)
val_loader = data.DataLoader(val_dataset, 
                                num_workers=4, batch_size=flags.batch_size, 
                                shuffle=False)
test_loader = data.DataLoader(test_dataset, 
                                num_workers=4, batch_size=flags.batch_size, 
                                shuffle=False)


def to_device(data):
    for i in range(len(data)):
        data[i] = data[i].to(device)
    return data

best_by_val = 0
best_val_acc = 0.0
best_val_loss = float('inf')
for epoch in range(flags.epochs):
    print('Epoch {}: Best by val {}'.format(epoch,best_by_val))
    lossMeter = AverageMeter()
    regMeter = AverageMeter()
    correctMeter = AverageMeter()
    model.train()
    for data in tqdm(train_loader,ncols=75,leave=False):
        data = to_device(data)
        loss, reg, correct = model(*data)

        obj = loss + reg

        optim.zero_grad()
        obj.backward()
        optim.step()

        lossMeter.update(loss.item(),data[0].shape[0])
        regMeter.update(reg.item(),data[0].shape[0])
        correctMeter.update(correct.item(),data[0].shape[0])
        del loss, reg, correct
    print('>>> Training: Loss ', lossMeter,', Reg ', regMeter,', Acc ', correctMeter)

    vallossMeter = AverageMeter()
    valregMeter = AverageMeter()
    valcorrectMeter = AverageMeter()
    model.eval()
    with torch.no_grad():
        for x,y in tqdm(val_loader,ncols=75,leave=False):
            x,y = x.to(device), y.to(device)
            loss, reg, correct = model(x,y)

            vallossMeter.update(loss.item(),x.shape[0])
            valregMeter.update(reg.item(),x.shape[0])
            valcorrectMeter.update(correct.item(),x.shape[0])
            del loss, reg, correct
    print('>>> Val: Loss ', vallossMeter,', Reg ', valregMeter,', Acc ', valcorrectMeter)


    testlossMeter = AverageMeter()
    testregMeter = AverageMeter()
    testcorrectMeter = AverageMeter()
    model.eval()
    with torch.no_grad():
        for x,y in tqdm(test_loader,ncols=75,leave=False):
            x,y = x.to(device), y.to(device)
            loss, reg, correct = model(x,y)

            testlossMeter.update(loss.item(),x.shape[0])
            testregMeter.update(reg.item(),x.shape[0])
            testcorrectMeter.update(correct.item(),x.shape[0])
            del loss, reg, correct
    print('>>> Test: Loss ', testlossMeter,', Reg ', testregMeter,', Acc ', testcorrectMeter)


    if vallossMeter.float()<best_val_loss and valcorrectMeter.float()>best_val_acc:
        best_by_val = testcorrectMeter.float()
        torch.save(model.state_dict(),'saved/{}_{}_target{}_seed{}.pt'.format(flags.dataset,flags.model,flags.target_domain,flags.seed))
    if vallossMeter.float()<best_val_loss:
        best_val_loss = vallossMeter.float()
    if valcorrectMeter.float()>best_val_acc:
        best_val_acc = valcorrectMeter.float()




