import numpy as np
import torch
from torch.utils.data.dataset import T_co
import torchvision.transforms as transforms

import torch.utils.data as data

from PIL import Image
from torchvision.datasets import CIFAR10

from ..baseDataset import dataset
from ..partition.partitionFactory import partitionFactory
from core.log.Log import Log


class cifar10Controller(dataset):
    def __init__(self, parse, transform=None):
        self.parse = parse
        self.target_transform = None
        self.transform = transform
        self.data = None
        self.log = Log(self.__class__.__name__)
        self.root = parse['dataDir']
        self.target = None
        # self.n_nets = parse['client_number'] if parse['client_number'] else 16
        self.train = parse['train'] if parse['train'] is not None else True
        self.dataidxs = parse['dataidxs']
        self.download = parse['download'] if parse['download'] is not None else False
        self.data, self.target = self.__build_truncated_dataset__()

    def __getitem__(self, index) -> T_co:
        """
                Args:
                    index (int): Index

                Returns:
                    tuple: (image, target) where target is index of the target class.
         """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

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

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def loadData(self):
        """
        加载数据
        """
        self.log.info(self.parse.dataDir)
        train_transform, test_transform = _data_transforms_cifar10()

        cifar10_train_ds = cifar10Controller(parse=self.parse, transform=train_transform)
        cifar10_test_ds = cifar10Controller(parse=self.parse, transform=test_transform)

        X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
        X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

        return X_train, y_train, X_test, y_test

    def partition_data(self):
        partition_method = partitionFactory(parse=self.parse).factory()

        self.log.info(partition_method)
        return partition_method(self.parse, self.loadData)


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform
