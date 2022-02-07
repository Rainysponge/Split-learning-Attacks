import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from core.log.Log import Log


class adult_truncated(data.Dataset):
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
        adult_dataobj = pd.read_csv('{}/adult.data'.format(self.root), header=None)
        row = adult_dataobj[0:1]
        # self.log.info(adult_dataobj[7])
        d = {}
        for index in row:
            temp = row[index]
            if temp.dtype != int:
                keys = list(set(adult_dataobj[index]))
                values = range(len(keys))
                d.update(dict(zip(keys, values)))

        train = adult_dataobj.applymap(lambda x: d[x] if type(x) != int else x)
        # data.to_csv('./data/adult/PreProcess_adult.data', header=None, index=None)
        d.update({' <=50K.': d[' <=50K'], ' >50K.': d[' >50K']})

        # adult_dataobj = pd.read_csv('./data/adult/adult.data', header=None)

        adult_test_obj = pd.read_csv('{}/adult.test'.format(self.root), header=None)
        test = adult_test_obj.applymap(lambda x: d[x] if type(x) != int else x)

        train = np.array(train)
        test = np.array(test)

        n, l = train.shape
        for j in range(l - 1):
            if self.parse["partition_method"] == "base_on_attribute":
                if self.parse["partition_method_attributes"] == j:
                    t = len(np.unique(train[:, j])) // 2
                    train[:, j] = train[:, j] - t
                    continue
            meanVal = abs(np.mean(train[:, j]))
            stdVal = np.std(train[:, j])
            train[:, j] = (train[:, j] - meanVal) / stdVal

        # np.random.shuffle(train)
        n, l = test.shape
        for j in range(l - 1):
            meanVal = np.mean(test[:, j])
            stdVal = np.std(test[:, j])
            test[:, j] = (test[:, j] - meanVal) / stdVal
            # t = len(np.unique(test[:, j])) // 2
            # test[:, j] = test[:, j] - t
        np.random.shuffle(test)

        train_data = train[:, :14]
        train_lab = train[:, 14]

        test_data = test[:, :14]
        test_lab = test[:, 14]
        if self.train:
            data = torch.from_numpy(train_data).float()
            target = train_lab
        else:
            data = torch.from_numpy(test_data).float()
            target = test_lab

        # # 归一化
        # n, l = train.shape
        # for j in range(l - 1):
        #     meanVal = np.mean(train[:, j])
        #     stdVal = np.std(train[:, j])
        #     train[:, j] = (train[:, j] - meanVal) / stdVal
        #
        #
        # n, l = test.shape
        # for j in range(l - 1):
        #     meanVal = np.mean(test[:, j])
        #     stdVal = np.std(test[:, j])
        #     test[:, j] = (test[:, j] - meanVal) / stdVal



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
