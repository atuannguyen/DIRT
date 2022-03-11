import torchvision.models as models
import torch
from torch import nn
import torch.nn.functional as F
from models.base import BaseModel
import pdb

class Model(BaseModel):
    def __init__(self, hidden_dim=256, base='resnet50'):
        super(Model, self).__init__(hidden_dim,base)

        #self.w1 = nn.Parameter(torch.ones([hidden_dim,7]))
        #torch.nn.init.kaiming_normal_(self.w1)
        self.out_layer = nn.Linear(hidden_dim,7)

    def forward(self, x,y):
        #x = x.permute(0,3,1,2).contiguous()
        z = F.relu(self.base(x))
        #logits = torch.mm(x,self.w1)
        logits = self.out_layer(z)
        loss = F.cross_entropy(logits,y)
        if self.training:
            loss = loss + F.mse_loss(z,z.new_zeros(z.shape))
        correct = (torch.argmax(logits,1)==y).sum().float()/x.shape[0]
        reg =loss.new_zeros([1])
        return loss,reg,correct

