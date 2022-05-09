import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from core.log.Log import Log


class german_truncated(data.Dataset):
    def __init__(self, parse, dataidxs=None, train=True, transform=None):

        self.parse = parse
        self.target_transform = None
        self.transform = transform
        self.data = None
        self.dataidxs = dataidxs
        self.log = Log(self.__class__.__name__, parse)
        self.root = parse['dataDir']+parse['dataset']
        self.target = None
        self.bantch_size = parse["batch_size"]
        # self.n_nets = parse['client_number'] if parse['client_number'] else 16
        self.train = train

        self.download = parse['download'] if parse['download'] is not None else False

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        data = np.loadtxt("./data/german/german.data-numeric")
        n, l = data.shape
        for j in range(l - 1):
            meanVal = np.mean(data[:, j])
            stdVal = np.std(data[:, j])
            data[:, j] = (data[:, j] - meanVal) / stdVal
        # np.random.shuffle(data)
        train_data = data[:900, :l - 1]
        train_lab = data[:900, l - 1] - 1
        test_data = data[900:, :l - 1]
        test_lab = data[900:, l - 1] - 1

        if self.train:
            data = torch.from_numpy(train_data).float()
            target = train_lab
        else:
            data = torch.from_numpy(test_data).float()
            target = test_lab

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target


    def truncate_channel(self, index):
        pass
        # for i in range(index.shape[0]):
        #     gs_index = index[i]
        #     self.data[gs_index, :, :, 1] = 0.0
        #     self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        data, target = self.data[index], int(self.target[index])
        return data, target
        # img, target =
        #
        # # doing this so that it is consistent with all other datasets
        # # to return a PIL Image
        # img = Image.fromarray(img.numpy(), mode='L')
        #
        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        #
        # return img, target

    def __len__(self):
        return len(self.data)
