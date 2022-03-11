import torchvision.models as models
import torch
from torch import nn
import torch.nn.functional as F
from models.base import BaseModel

class Model(BaseModel):
    def __init__(self, hidden_dim=256, base='resnet50'):
        super(Model, self).__init__(hidden_dim,base)

        self.w1_mu = nn.Parameter(torch.ones([hidden_dim,7]))
        self.w1_sigma = nn.Parameter(torch.ones([hidden_dim,7]))
        torch.nn.init.kaiming_normal_(self.w1_mu)


    def forward(self, x,y):
        x = x.permute(0,3,1,2).contiguous()
        pdb.set_trace()
        x = F.relu(self.base(x))
        w1_dist = torch.distributions.normal.Normal(self.w1_mu,self.w1_sigma)
        if self.training:
            w1 = w1_dist.rsample()
            logits = torch.mm(x,w1)
            loss = F.cross_entropy(logits,y)
            correct = (torch.argmax(logits,1)==y).sum().float()/x.shape[0]
            KL = -0.5 * (1+torch.log(self.w1_sigma**2)-self.w1_mu**2-self.w1_sigma**2).sum()
            reg = KL/7000
        else:
            preds = 0
            for i in range(10):
                w1 = w1_dist.sample()
                preds += F.softmax(torch.mm(x,w1),1)
            preds = preds / 10
            loss = preds.new_zeros([1])
            reg = preds.new_zeros([1])
            correct = (torch.argmax(preds,1)==y).sum().float()/x.shape[0]
        return loss,reg,correct

