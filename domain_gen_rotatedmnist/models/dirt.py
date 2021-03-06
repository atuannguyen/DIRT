import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.stargan import *



class Model(nn.Module):
    def __init__(self,config):
        super(Model,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.fc11 = nn.Sequential(nn.Linear(1024, 64))

        torch.nn.init.xavier_uniform_(self.encoder[0].weight)
        torch.nn.init.xavier_uniform_(self.encoder[4].weight)
        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()

        self.cls = nn.Linear(64, 10)
        torch.nn.init.xavier_uniform_(self.cls.weight)
        self.cls.bias.data.zero_()
        
        self.trans = load_stargan(ckpt='saved/stargan_model/{}_domain{}_last-G.ckpt'.format(config.dataset,config.target_domain)) 
        self.trans.eval()

    def forward(self,x,y,d=None):
        h = self.encoder(x)
        h = h.view(-1, 1024)
        z = self.fc11(h)

        logits = self.cls(F.relu(z))
        loss = F.cross_entropy(logits, y)
        acc = ((logits.argmax(1)==y).sum().float()/len(y)).item()

        if self.training:
            with torch.no_grad():
                one_hot_d = x.new_zeros([x.shape[0],5])
                one_hot_d.scatter_(1, d[:,None], 1)
                d_ = x.new_ones(x.shape[0]).to(torch.int64)*np.random.choice(5)
                one_hot_d_ = x.new_zeros([x.shape[0],5])
                one_hot_d_.scatter_(1, d_[:,None], 1)
                x_ = self.trans(x,one_hot_d,one_hot_d_)
            h_ = self.encoder(x_)
            h_ = h_.view(-1, 1024)
            z_ = self.fc11(h_)
            reg = F.mse_loss(z_,z,'mean')
            loss = loss + reg

        return loss, acc
