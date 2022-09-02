import logging
import torch
import numpy as np
import torch.utils.data as data
import pandas as pd
import scipy.sparse as sp
from torch.nn.functional import normalize
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from torch_geometric.data import Data, Dataset
from torch_geometric.data.data import size_repr
# from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_sum
from tqdm import tqdm


from PIL import Image

from core.log.Log import Log


class RevIndexedData(Data):
    def __init__(self, orig):
        super(RevIndexedData, self).__init__()
        if orig:
            for key in orig.keys:
                self[key] = orig[key]
            edge_index = self["edge_index"]
            revedge_index = torch.zeros(edge_index.shape[1]).long()
            for k, (i, j) in enumerate(zip(*edge_index)):
                edge_to_i = edge_index[1] == i
                edge_from_j = edge_index[0] == j
                revedge_index[k] = torch.where(edge_to_i & edge_from_j)[0].item()
            self["revedge_index"] = revedge_index

    def __inc__(self, key, value, *args, **kwargs):
        if key == "revedge_index":
            return self.revedge_index.max().item() + 1
        else:
            return super().__inc__(key, value)

    def __repr__(self):
        cls = str(self.__class__.__name__)
        has_dict = any([isinstance(item, dict) for _, item in self])

        if not has_dict:
            info = [size_repr(key, item) for key, item in self]
            return "{}({})".format(cls, ", ".join(info))
        else:
            info = [size_repr(key, item, indent=2) for key, item in self]
            return "{}(\n{}\n)".format(cls, ",\n".join(info))


class RevIndexedDataset(Dataset):
    def __init__(self, orig):
        super(RevIndexedDataset, self).__init__()
        self.dataset = [RevIndexedData(data) for data in tqdm(orig)]

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


def directed_mp(message, edge_index, revedge_index):
    m = scatter_sum(message, edge_index[1], dim=0)
    m_all = m[edge_index[0]]
    m_rev = message[revedge_index]
    return m_all - m_rev


def aggregate_at_nodes(num_nodes, message, edge_index):
    m = scatter_sum(message, edge_index[1], dim=0, dim_size=num_nodes)
    return m[torch.arange(num_nodes)]


class molhiv_truncated(data.Dataset):

    def __init__(self, parse, dataidxs=None, train=True, transform=None):

        self.parse = parse
        self.target_transform = None
        self.transform = transform
        self.data = None
        self.dataidxs = dataidxs
        self.log = Log(self.__class__.__name__, parse)
        self.root = parse['dataDir'] + "molhiv/"
        self.target = None
        self.bantch_size = parse["batch_size"]
        # self.n_nets = parse['client_number'] if parse['client_number'] else 16
        self.train = train

        self.download = parse['download'] if parse['download'] is not None else False
        self.pyg_dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=self.root)
        self.datas = RevIndexedDataset(self.pyg_dataset)
        # self.data, self.adj, self.idx = self.__build_truncated_dataset__()
        self.data, self.target = self.__build_truncated_dataset__()


    def __build_truncated_dataset__(self):
        # pyg_dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=self.root)
        split_idx = self.pyg_dataset.get_idx_split()

        y_list = []
        if self.train:
            data = torch.utils.data.Subset(self.datas, split_idx["train"])
        else:
            data = torch.utils.data.Subset(self.datas, split_idx["test"])

        if self.dataidxs is not None:
            data = torch.utils.data.Subset(self.datas, self.dataidxs)
        for i in range(len(data)):
            y_list.append(data[i].y.float()[0][0])
        return data, np.array(y_list)



    def get_subdataset(self, dataidxs=None, train=True):
        split_idx = self.pyg_dataset.get_idx_split()


        if train:
            data = torch.utils.data.Subset(self.datas, split_idx["train"])
        else:
            data = torch.utils.data.Subset(self.datas, split_idx["test"])

        if self.dataidxs is not None:
            data = torch.utils.data.Subset(self.datas, dataidxs)

        return data

    def __getitem__(self, index):
        return self.data[index], self.data[index].y.float()

    def __len__(self):
        return len(self.data)
