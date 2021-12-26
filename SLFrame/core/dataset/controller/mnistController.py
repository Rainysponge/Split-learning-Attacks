import numpy as np
import torch
from torch.utils.data.dataset import T_co
import torchvision.transforms as transforms

import torch.utils.data as data

from PIL import Image
from torchvision.datasets import MNIST

from ..baseDataset import dataset
from ..partition.partitionFactory import partitionFactory
from core.log.Log import Log


class mnistController(dataset):
    def __init__(self, parse, transform=None):
        self.parse = parse
        self.target_transform = None
        self.transform = transform
        self.data = None
        self.log = Log(self.__class__.__name__)
        self.root = parse['dataDir']
        self.target = None
        self.bantch_size = parse["batch_size"]
        # self.n_nets = parse['client_number'] if parse['client_number'] else 16
        self.train = parse['train'] if parse['train'] is not None else True
        self.dataidxs = parse['dataidxs']
        self.download = parse['download'] if parse['download'] is not None else False
        self.data, self.target = self.__build_truncated_dataset__()

    def loadData(self):
        self.log.info(self.parse.dataDir)
        train_transform, test_transform = _data_transforms_mnist()

        mnist_train_ds = mnistController(parse=self.parse, transform=train_transform)
        mnist_test_ds = mnistController(parse=self.parse, transform=test_transform)

        X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
        X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

        return X_train, y_train, X_test, y_test

    def partition_data(self):
        partition_method = partitionFactory(parse=self.parse).factory()

        self.log.info(partition_method)
        return partition_method(self.loadData)

    def get_dataloader(self):
        dl_obj = mnistController

        transform_train, transform_test = _data_transforms_mnist()

        train_ds = dl_obj(parse=self.parse, transform=transform_train)
        test_ds = dl_obj(parse=self.parse, transform=transform_test)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=self.bantch_size, shuffle=True, drop_last=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=self.bantch_size, shuffle=False, drop_last=True)

        return train_dl, test_dl

    def load_partition_data(self, process_id):
        X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = self.partition_data()
        class_num = len(np.unique(y_train))
        self.log.info("traindata_cls_counts = " + str(traindata_cls_counts))
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
            # training batch size = 64;
            train_data_local, test_data_local = self.get_dataloader()
            self.log.info("process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
                process_id, len(train_data_local), len(test_data_local)))
            train_data_global = None
            test_data_global = None
        return train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num

    def __getitem__(self, index) -> T_co:

        img, target = self.data[index], int(self.target[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

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
        cifar_dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)

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


def _data_transforms_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    return transform, transform
