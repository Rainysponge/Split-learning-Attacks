from torch.utils.data.dataset import T_co
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import CIFAR10

from ..abstractDataset import dataset
from core.log.Log import Log



class cifar10Controller(dataset):
    def __init__(self, parse):
        self.target_transform = None
        self.transform = None
        self.target = None
        self.data = None
        self.log = Log(self.__class__.__name__)
        self.parse = parse

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

    def loadData(self):
        # self.log.info(self.parse.dataDir)
        pass

    def partition_data(self, dataset, partition, n_nets, alpha):
        pass
