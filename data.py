import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


class LossWriter:
    def __init__(self, save_path):
        self.save_path = save_path

    def add(self, loss, i):
        with open(self.save_path, mode="a") as f:
            term = str(i) + " " + str(loss) + "\n"
            f.write(term)
            f.close()


def loadMNIST(img_size, batch_size):  # MNIST图片的大小是28*28
    trans_img = transforms.Compose(
        [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    trainset = MNIST('./data', train=True, transform=trans_img, download=True)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    return trainloader


def onehot(label, num_class):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = label.shape[0]
    onehot_label = torch.zeros(n, num_class, dtype=label.dtype).to(device)
    onehot_label = onehot_label.scatter_(1, label.view(n, 1), 1)
    return onehot_label
