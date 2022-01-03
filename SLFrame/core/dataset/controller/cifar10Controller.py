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
from ..dataset.cifar10 import cifar10_truncated

class cifar10Controller():
    def __init__(self, parse, transform=None):
        self.parse = parse
        self.target_transform = None
        self.transform = transform
        self.data = None
        self.log = Log(self.__class__.__name__, parse)
        self.root = parse['dataDir']
        self.target = None
        self.bantch_size = parse["batch_size"]
        # self.n_nets = parse['client_number'] if parse['client_number'] else 16
        self.train = parse['train'] if parse['train'] is not None else True
        self.dataidxs = parse['dataidxs']
        self.download = parse['download'] if parse['download'] is not None else False

    def loadData(self):
        self.log.info(self.parse.dataDir)
        train_transform, test_transform = _data_transforms_cifar10()

        cifar10_train_ds = cifar10_truncated(parse=self.parse, transform=train_transform)
        cifar10_test_ds = cifar10_truncated(parse=self.parse, transform=test_transform)

        X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
        X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

        return X_train, y_train, X_test, y_test

    def partition_data(self):
        partition_method = partitionFactory(parse=self.parse).factory()

        self.log.info(partition_method)
        return partition_method(self.loadData)

    def get_dataloader_CIFAR10(self, dataidxs=None):
        dl_obj = cifar10_truncated

        transform_train, transform_test = _data_transforms_cifar10()

        train_ds = dl_obj(parse=self.parse, transform=transform_train, dataidxs=dataidxs)
        test_ds = dl_obj(parse=self.parse, transform=transform_test, train=False)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=self.bantch_size, shuffle=True, drop_last=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=self.bantch_size, shuffle=False, drop_last=True)

        return train_dl, test_dl

    def get_dataloader(self, dataidxs=None):
        return self.get_dataloader_CIFAR10(dataidxs)

    def load_partition_data(self, process_id):
        X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = self.partition_data()
        class_num = len(np.unique(y_train))
        # self.log.info("traindata_cls_counts = " + str(traindata_cls_counts))
        train_data_num = sum([len(net_dataidx_map[r]) for r in range(self.parse["client_number"])])

        # get global test data
        if process_id == 0:
            train_data_global, test_data_global = self.get_dataloader()
            self.log.info("train_dl_global number = " + str(len(train_data_global)))
            self.log.info("test_dl_global number = " + str(len(test_data_global)))
            train_data_local = None
            test_data_local = None
            local_data_num = 0
        else:
            # get local dataset
            dataidxs = net_dataidx_map[process_id - 1]
            local_data_num = len(dataidxs)
            self.log.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
            # training batch size = 64; algorithms batch size = 32
            train_data_local, test_data_local = self.get_dataloader(dataidxs)
            self.log.info("process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
                process_id, len(train_data_local), len(test_data_local)))
            train_data_global = None
            test_data_global = None
        return train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num


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
