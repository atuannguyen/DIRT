import torchvision.models as models
import torch
from torch import nn
import torch.nn.functional as F
from models.base import BaseModel
import numpy as np

class Model(BaseModel):
    def __init__(self, hidden_dim=256, base='resnet50'):
        super(Model, self).__init__(hidden_dim,base)

        self.w1_mu = nn.Parameter(torch.ones([7,hidden_dim]))
        self.w1_sigma = nn.Parameter(torch.ones([7,hidden_dim]))
        self.b1_mu = nn.Parameter(torch.zeros([7]))
        self.b1_sigma = nn.Parameter(torch.ones([7]))
        torch.nn.init.kaiming_normal_(self.w1_mu,mode='fan_out')


    def forward(self, x,y):
        x = x.permute(0,3,1,2).contiguous()
        x = F.relu(self.base(x))
        w1_dist = torch.distributions.normal.Normal(self.w1_mu,self.w1_sigma)
        b1_dist = torch.distributions.normal.Normal(self.b1_mu,self.b1_sigma)
        if self.training:
            w1 = w1_dist.rsample()
            b1 = b1_dist.rsample()
            logits = F.linear(x,w1,b1)
            loss = F.cross_entropy(logits,y)
            correct = (torch.argmax(logits,1)==y).sum().float()/x.shape[0]
            H = 0.5 * torch.log(2*np.pi*np.e*self.w1_sigma**2).sum() +\
                    0.5 * torch.log(2*np.pi*np.e*self.b1_sigma**2).sum()
            reg = -H/3000
        else:
            preds = 0
            for i in range(10):
                w1 = w1_dist.sample()
                b1 = b1_dist.sample()
                preds += F.softmax(F.linear(x,w1,b1),1)
            preds = preds / 10
            loss = F.nll_loss(torch.log(preds),y)
            reg = preds.new_zeros([1])
            correct = (torch.argmax(preds,1)==y).sum().float()/x.shape[0]
        return loss,reg,correct

