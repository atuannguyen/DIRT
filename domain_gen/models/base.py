import torchvision.models as models
import torch
from torch import nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self, hidden_dim=256, base='resnet50'):
        super(BaseModel, self).__init__()
        if base=='alexnet':
            self.base = models.alexnet(pretrained=True)
            self.base.classifier[6] = nn.Linear(self.base.classifier[6].in_features, hidden_dim)
        elif base=='resnet50':
            self.base = models.resnet50(pretrained=True)
            self.base.fc = nn.Linear(self.base.fc.in_features, hidden_dim)
        elif base=='resnet18':
            self.base = models.resnet18(pretrained=True)
            self.base.fc = nn.Linear(self.base.fc.in_features, hidden_dim)


