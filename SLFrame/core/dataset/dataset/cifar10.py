import logging

import numpy as np
import torch
from numpy.compat import long
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import CIFAR10

from core.log.Log import Log


class cifar10_truncated(data.Dataset):

    def __init__(self, parse, dataidxs=None, train=True, transform=None):

        self.parse = parse
        self.target_transform = None
        self.transform = transform
        self.data = None
        self.dataidxs = dataidxs
        self.log = Log(self.__class__.__name__, parse)
        self.root = parse['dataDir']
        self.target = None
        self.bantch_size = parse["batch_size"]
        # self.n_nets = parse['client_number'] if parse['client_number'] else 16
        self.train = train

        self.download = parse['download'] if parse['download'] is not None else False

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        # print("download = " + str(self.download))
        # Files already downloaded and verified日志来自这里
        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        target = torch.Tensor(target).long()  # Factos!!!
        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):

        img, target = self.data[index], self.target[index]
        # self.log.info(type(target.long()))
        # target = Image.fromarray(target, mode='L')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
