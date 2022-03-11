import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mnist_loader import MnistRotated


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, img_channels=1):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(img_channels+2*c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, img_channels, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c_org, c_trg):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c_org = c_org.view(c_org.size(0), c_org.size(1), 1, 1)
        c_org = c_org.repeat(1, 1, x.size(2), x.size(3))
        c_trg = c_trg.view(c_trg.size(0), c_trg.size(1), 1, 1)
        c_trg = c_trg.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c_org, c_trg], dim=1)
        return self.main(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6, img_channels=1):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(img_channels, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2


        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))

def load_stargan(ckpt='saved/stargan.pt'):
    g = Generator(64,5,6)
    g.load_state_dict(torch.load(ckpt))
    return g

if __name__=='__main__':
    g = Generator(64,5,6)
    g.load_state_dict(torch.load('stargan/models/80000-G.ckpt'))
    d = Discriminator(28,64,5,4)

    dataset = MnistRotated(['0','15','30','45','60'],['75'],1000,0,'dataset/', train=True,only_domain_label=True)
    data_loader = DataLoader(data_set,batch_size=64,shuffle=False)
    x,d = next(iter(data_loader))
    
    import matplotlib.pyplot as plt
    x_t = [None]*5
    x_r = [None]*5
    for j in range(5):
        d_ = x.new_ones(x.shape[0]).to(torch.int64)*j
        one_hot_d_ = x.new_zeros([x.shape[0],5])
        one_hot_d_.scatter_(1, d_[:,None], 1)
        x_t = g(x,one_hot_d_)
        import pdb; pdb.set_trace()
        
    for i in range(10):
        fig=plt.figure()
        fig.add_subplot(1, 11, 1)
        plt.imshow(x[i][0].cpu().data, cmap='gray')
        plt.axis('off')
    import pdb; pdb.set_trace()

