import torch
from torch import optim
from torch import nn
from data import LossWriter, loadMNIST, onehot
from model import Generator, Discriminator

DATA_DIR = "../datasets/selfie2anime/all"
MODEL_G_PATH = "./Net_G.pth"
MODEL_D_PATH = "./Net_D.pth"
LOG_G_PATH = "Log_G.txt"
LOG_D_PATH = "Log_D.txt"
IMAGE_SIZE = 64
BATCH_SIZE = 128
WORKER = 1
LR = 0.0002
NZ = 100
NUM_CLASS = 10
EPOCH = 300

data_loader = loadMNIST(img_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)
netD = Discriminator().to(device)
criterion = nn.BCELoss()
real_label = 1.
fake_label = 0.
optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.999))

g_writer = LossWriter(save_path=LOG_G_PATH)
d_writer = LossWriter(save_path=LOG_D_PATH)

img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
for epoch in range(EPOCH):
    for data in data_loader:
        #################################################
        # 1. 更新判别器D: 最大化 log(D(x)) + log(1 - D(G(z)))
        # 等同于最小化 - log(D(x)) - log(1 - D(G(z)))
        #################################################
        netD.zero_grad()
        real_imgs, input_c = data
        input_c = input_c.to(device)
        input_c = onehot(input_c, NUM_CLASS).to(device)

        # 1.1 来自数据集的样本
        real_imgs = real_imgs.to(device)
        b_size = real_imgs.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # 使用鉴别器对数据集样本做判断
        output = netD(real_imgs, input_c).view(-1)
        # 计算交叉熵损失 -log(D(x))
        errD_real = criterion(output, label)
        # 对判别器进行梯度回传
        errD_real.backward()
        D_x = output.mean().item()

        # 1.2 生成随机向量
        noise = torch.randn(b_size, NZ, device=device)
        # 生成随机标签
        input_c = (torch.rand(b_size, 1) * NUM_CLASS).type(torch.LongTensor).squeeze().to(device)
        input_c = onehot(input_c, NUM_CLASS)
        # 来自生成器生成的样本
        fake = netG(noise, input_c)
        label.fill_(fake_label)
        # 使用鉴别器对生成器生成样本做判断
        output = netD(fake.detach(), input_c).view(-1)
        # 计算交叉熵损失 -log(1 - D(G(z)))
        errD_fake = criterion(output, label)
        # 对判别器进行梯度回传
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # 对判别器计算总梯度,-log(D(x))-log(1 - D(G(z)))
        errD = errD_real + errD_fake
        # 更新判别器
        optimizerD.step()

        #################################################
        # 2. 更新判别器G: 最小化 log(D(x)) + log(1 - D(G(z)))，
        # 等同于最小化log(1 - D(G(z)))，即最小化-log(D(G(z)))
        # 也就等同于最小化-（log(D(G(z)))*1+log(1-D(G(z)))*0）
        # 令生成器样本标签值为1，上式就满足了交叉熵的定义
        #################################################
        netG.zero_grad()
        # 对于生成器训练，令生成器生成的样本为真，
        label.fill_(real_label)
        # 输入生成器的生成的假样本
        output = netD(fake, input_c).view(-1)
        # 生成随机标签
        input_c = (torch.rand(b_size, 1) * NUM_CLASS).type(torch.LongTensor).squeeze().to(device)
        input_c = onehot(input_c, NUM_CLASS)
        # 对生成器计算损失
        errG = criterion(output, label)
        # 对生成器进行梯度回传
        errG.backward()
        D_G_z2 = output.mean().item()
        # 更新生成器
        optimizerG.step()

        # 输出损失状态
        if iters % 5 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, EPOCH, iters, len(data_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            d_writer.add(loss=errD.item(), i=iters)
            g_writer.add(loss=errG.item(), i=iters)

        # 保存损失记录
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        iters += 1
