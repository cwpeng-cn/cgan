import torch
import numpy as np
import pylab as plt
from model import Generator
from data import restore_network, onehot, recover_image

NZ = 100
NUM_CLASS = 10
BATCH_SIZE = 10
DEVICE = "cpu"

# fix_input_c = (torch.rand(BATCH_SIZE, 1) * NUM_CLASS).type(torch.LongTensor).squeeze().to(DEVICE)

netG = Generator()
netG = restore_network("./", "last", netG)
fix_noise = torch.randn(BATCH_SIZE, NZ, device=DEVICE)
fix_input_c = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
fix_input_c = onehot(fix_input_c, NUM_CLASS)
fake_imgs = netG(fix_noise, fix_input_c).detach().cpu()

# images = recover_image(fake_imgs)
# full_image = np.full((1 * 64, 10 * 64, 3), 0, dtype="uint8")
# for i in range(10):
#     row = i // 10
#     col = i % 10
#     full_image[row * 64:(row + 1) * 64, col * 64:(col + 1) * 64, :] = images[i]

fix_noise = torch.randn(BATCH_SIZE, NZ, device=DEVICE)
full_image = np.full((10 * 64, 10 * 64, 3), 0, dtype="uint8")
for num in range(10):
    input_c = torch.tensor(np.ones(10, dtype="int64") * num)
    input_c = onehot(input_c, NUM_CLASS)
    fake_imgs = netG(fix_noise, input_c).detach().cpu()
    images = recover_image(fake_imgs)
    for i in range(10):
        row = num
        col = i % 10
        full_image[row * 64:(row + 1) * 64, col * 64:(col + 1) * 64, :] = images[i]

plt.imshow(full_image)
plt.show()
plt.imsave("hah.png", full_image)
