import torch
import numpy as np
import pylab as plt
from model import Generator
from data import restore_network, onehot, recover_image

NZ = 100
NUM_CLASS = 10
BATCH_SIZE = 25
DEVICE = "cpu"

netG = Generator()
netG = restore_network("./", 0, netG)
fix_noise = torch.randn(BATCH_SIZE, NZ, device=DEVICE)
fix_input_c = (torch.rand(BATCH_SIZE, 1) * NUM_CLASS).type(torch.LongTensor).squeeze().to(DEVICE)
fix_input_c = onehot(fix_input_c, NUM_CLASS)
fake_imgs = netG(fix_noise, fix_input_c).detach().cpu()

images = recover_image(fake_imgs)
full_image = np.full((5 * 64, 5 * 64, 3), 0, dtype="uint8")
for i in range(25):
    row = i // 5
    col = i % 5
    full_image[row * 64:(row + 1) * 64, col * 64:(col + 1) * 64, :] = images[i]

plt.imshow(full_image)
plt.show()
plt.imsave("hah.png", full_image)
