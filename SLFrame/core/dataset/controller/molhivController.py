import numpy as np
import torch
from torch.utils.data.dataset import T_co
import torchvision.transforms as transforms

import torch.utils.data as data
from torch_geometric.loader import DataLoader
from PIL import Image

from ..dataset.molhiv import molhiv_truncated
from ..partition.partitionFactory import partitionFactory
from core.log.Log import Log


class molhivController():
    def __init__(self, parse, transform=None):
        self.parse = parse
        self.target_transform = None
        self.transform = transform
        self.data = None
        self.log = Log(self.__class__.__name__, parse)
        # self.root1 = parse['dataDir'] + "molhiv/"
        # self.pyg_dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=self.root1)
        self.root = parse['dataDir']
        self.target = None
        self.bantch_size = parse["batch_size"]
        # self.n_nets = parse['client_number'] if parse['client_number'] else 16
        self.train = parse['train'] if parse['train'] is not None else True
        self.download = parse['download'] if parse['download'] is not None else False

    def loadData(self):
        self.log.info(self.parse.dataDir)
        # train_transform, test_transform = _data_transforms_mnist()

        self.molhiv_train_ds = molhiv_truncated(parse=self.parse)
        self.molhiv_test_ds = molhiv_truncated(parse=self.parse, train=False)

        X_train, y_train = self.molhiv_train_ds.data, self.molhiv_train_ds.target
        X_test, y_test = self.molhiv_test_ds.data, self.molhiv_test_ds.target

        return X_train, y_train, X_test, y_test

    def partition_data(self):
        partition_method = partitionFactory(parse=self.parse).factory()

        self.log.info(partition_method)
        return partition_method(self.loadData)

    def get_dataloader(self, dataidxs=None):

        # train_ds = dl_obj(parse=self.parse, transform=None, dataidxs=dataidxs)
        # test_ds = dl_obj(parse=self.parse, transform=None, train=False)
        if dataidxs is not None:
            train_dl = molhiv_dataloader(dataset=self.molhiv_train_ds.get_subdataset(dataidxs),
                                         batch_size=self.bantch_size, shuffle=True)
        else:
            train_dl = molhiv_dataloader(dataset=self.molhiv_train_ds.get_subdataset(),
                                         batch_size=self.bantch_size, shuffle=True)
        test_dl = molhiv_dataloader(dataset=self.molhiv_test_ds.get_subdataset(train=False),
                                    batch_size=self.bantch_size,
                                    shuffle=False)

        return train_dl, test_dl

    def load_partition_data(self, process_id):
        X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = self.partition_data()
        class_num = len(np.unique(y_train))
        # self.log.info("class_num = " + str(class_num))
        train_data_num = sum([len(net_dataidx_map[r]) for r in net_dataidx_map.keys()])

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
            train_data_local, test_data_local = self.get_dataloader(dataidxs)
            # self.log.info("dataidxs: {}".format(dataidxs))
            self.log.info("process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
                process_id, len(train_data_local), len(test_data_local)))
            train_data_global = None
            test_data_global = None
        self.parse["trainloader"] = train_data_local
        self.parse["testloader"] = test_data_local
        self.parse["train_data_num"] = train_data_num
        self.parse["train_data_global"] = train_data_global
        self.parse["test_data_global"] = test_data_global
        self.parse["local_data_num"] = local_data_num
        self.parse["class_num"] = class_num
        return train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num


class molhiv_dataloader(DataLoader):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        self.loader = enumerate(self)

    def __next__(self):
        _, d = next(self.loader)
        return d, d.y.float()


