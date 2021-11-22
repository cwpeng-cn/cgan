import torch
from torch import nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, num_channel=1, nz=100, nc=10, ngf=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 输入维度 110 x 1 x 1
            nn.ConvTranspose2d(nz + nc, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 特征维度 (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 特征维度 (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 特征维度 (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 特征维度 (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, num_channel, 4, 2, 1, bias=False),
            nn.Tanh()
            # 特征维度. (num_channel) x 64 x 64
        )
        self.apply(weights_init)

    def forward(self, input_z, onehot_label):
        input_ = torch.cat((input_z, onehot_label), dim=1)
        n, c = input_.size()
        input_ = input_.view(n, c, 1, 1)
        return self.main(input_)


class Discriminator(nn.Module):
    def __init__(self, num_channel=1, nc=10, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 输入维度 (num_channel+nc) x 64 x 64
            nn.Conv2d(num_channel + nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 特征维度 (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 特征维度 (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 特征维度 (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 特征维度 (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, images, onehot_label):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        h, w = images.shape[2:]
        n, nc = onehot_label.shape[:2]
        label = onehot_label.view(n, nc, 1, 1) * torch.ones([n, nc, h, w]).to(device)
        input_ = torch.cat([images, label], 1)
        return self.main(input_)
