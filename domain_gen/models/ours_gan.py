import torchvision.models as models
import torch
from torch import nn
import torch.nn.functional as F
from models.base import BaseModel
import pdb
from models.stargan import load_stargan

class Model(BaseModel):
    def __init__(self, config, hidden_dim=256, base='resnet50'):
        super(Model, self).__init__(hidden_dim,base)

        #self.w1 = nn.Parameter(torch.ones([hidden_dim,7]))
        #torch.nn.init.kaiming_normal_(self.w1)
        self.out_layer = nn.Linear(hidden_dim,config.num_classes)
        self.trans = load_stargan(config.gan_path + '{}_domain{}_last-G.ckpt'.format(config.dataset,config.target_domain))
        self.trans.eval()

        self.alpha = config.alpha

    def forward(self, x,y,d=None):
        z = F.relu(self.base(x))
        #logits = torch.mm(x,self.w1)
        logits = self.out_layer(z)
        loss = F.cross_entropy(logits,y)
        correct = (torch.argmax(logits,1)==y).sum().float()/x.shape[0]
        reg =loss.new_zeros([1])
        if self.training:
            with torch.no_grad():
                rand_idx = torch.randperm(d.size(0))
                d_new = d[rand_idx]
                d_onehot = d.new_zeros([d.shape[0],3])
                d_onehot.scatter_(1, d[:,None], 1)
                d_new_onehot = d.new_zeros([d.shape[0],3])
                d_new_onehot.scatter_(1, d_new[:,None], 1)
                x_new = self.trans(x,d_onehot,d_new_onehot)

            z_new = F.relu(self.base(x_new))
            reg = self.alpha*F.mse_loss(z_new,z)

        return loss,reg,correct

