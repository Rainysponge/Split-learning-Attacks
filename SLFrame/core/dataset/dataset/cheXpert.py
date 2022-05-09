import numpy as np
import torch
import csv
from torch.utils.data import Dataset
import cv2
import os
from PIL import Image
import logging
from core.log.Log import Log

np.random.seed(0)


class cheXpert_truncated(Dataset):
    def __init__(self, parse, dataidxs=None, train=True, transform=None):
        self.parse = parse
        self.target_transform = None
        # self.transform = transform
        self.data = None
        self.dataidxs = dataidxs
        self.log = Log(self.__class__.__name__, parse)
        self.root = parse['dataDir']+ parse['dataset']
        self.target = None
        self.bantch_size = parse["batch_size"]
        # self.n_nets = parse['client_number'] if parse['client_number'] else 16
        self.train = train
        self.image_names = []
        self.labels = []
        self.dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
                     {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]

        self.download = parse['download'] if parse['download'] is not None else False

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        tmp = 0
        if self.train:
            csv_path = "data/CheXpert-v1.0-small/train.csv"
        else:
            csv_path = "data/CheXpert-v1.0-small/valid.csv"
        with open(csv_path) as f:
            header = f.readline().strip('\n').split(',')
            self._label_header = [
                header[7],
                header[10],
                header[11],
                header[13],
                header[15]]
            for line in f:
                labels = []
                fields = line.strip('\n').split(',')
                image_path = fields[0]
                flg_enhance = False
                for index, value in enumerate(fields[5:]):
                    if index == 5 or index == 8:
                        labels.append(self.dict[1].get(value))
                    elif index == 2 or index == 6 or index == 10:
                        labels.append(self.dict[0].get(value))
                # labels = ([self.dict.get(n, n) for n in fields[5:]])
                labels = list(map(int, labels))
                self.image_names.append("data/" + image_path)
                # logging.info("{} is loading data {} %".format(self.parse["rank"], tmp / 223414 * 100))
                tmp += 1
                assert os.path.exists("data/" + image_path), image_path
                self.labels.append(labels)

        self._num_image = len(self.image_names)
        self.image_shape = 256

        data = np.array(self.image_names)
        target = np.array(self.labels)
        # print(target)
        # target = np.array(target, dtype=np.long)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        target = torch.Tensor(target).long()
        return data, target[:, 1]

    def transfrom(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image = image.astype(np.float32) - 128.0
        image /= 64.0
        image = image.transpose((2, 0, 1))
        return image

    def __getitem__(self, index):
        index = index % 950
        image = cv2.imread(self.data[index], 0)
        image = cv2.resize(image, (self.image_shape, self.image_shape))
        image = Image.fromarray(image)

        image = np.array(image)
        image = self.transfrom(image)

        # labels = np.array(self.target[index]).astype(np.float32)
        labels = self.target[index]
        # labels = labels.long()
        # labels = torch.Tensor(labels).long()
        # path = self._image_paths[idx]

        # if self._mode == 'train' or self._mode == 'dev':
        #     return (image, labels)
        # else:
        #     raise Exception('Unknown mode : {}'.format(self._mode))

        return (image, labels)

    def __len__(self):
        return len(self.image_names)
