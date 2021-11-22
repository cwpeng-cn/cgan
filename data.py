import os
import torch
import numpy as np
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


def save_network(path, network, epoch_label, is_only_parameter=True):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join(path, save_filename)
    if is_only_parameter:
        state = network.state_dict()
        for key in state: state[key] = state[key].clone().cpu()
        torch.save(network.state_dict(), save_path, _use_new_zipfile_serialization=False)
    else:
        torch.save(network.cpu(), save_path)


def restore_network(path, epoch, network=None):
    path = os.path.join(path, 'net_%s.pth' % epoch)
    if network is None:
        network = torch.load(path)
    else:
        network.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return network


def recover_image(img):
    return (
            (img.numpy() *
             np.array([0.5]).reshape((1, 1, 1, 1)) +
             np.array([0.5]).reshape((1, 1, 1, 1))
             ).transpose(0, 2, 3, 1) * 255
    ).clip(0, 255).astype(np.uint8)
