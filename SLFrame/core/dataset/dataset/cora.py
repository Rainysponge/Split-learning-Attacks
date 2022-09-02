import logging
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from torch.nn.functional import normalize
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import MNIST

from core.log.Log import Log


class cora_truncated(data.Dataset):

    def __init__(self, parse, dataidxs=None, train=True, transform=None):

        self.parse = parse
        self.target_transform = None
        self.transform = transform
        self.data = None
        self.dataidxs = dataidxs
        self.log = Log(self.__class__.__name__, parse)
        self.root = parse['dataDir'] + "cora/"
        self.target = None
        self.bantch_size = parse["batch_size"]
        # self.n_nets = parse['client_number'] if parse['client_number'] else 16
        self.train = train

        self.download = parse['download'] if parse['download'] is not None else False

        self.data, self.adj, self.idx = self.__build_truncated_dataset__()

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def normalize_adj(self, adjacency):
        degree = np.array(adjacency.sum(1))
        d_hat = sp.diags(np.power(degree, -0.5).flatten())
        adj_norm = d_hat.dot(adjacency).dot(d_hat).tocoo()
        return adj_norm

    def normalize_features(self, features):
        return features / features.sum(1)

    def __build_truncated_dataset__(self):

        idx_features_labels = np.genfromtxt("{}{}.content".format(self.root, "cora"),
                                            dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = self.encode_onehot(idx_features_labels[:, -1])

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(self.root, "cora"),
                                        dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        features = self.normalize_features(features)
        adj = self.normalize_adj(adj + sp.eye(adj.shape[0]))

        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)

        features = torch.FloatTensor(np.array(features))
        labels = torch.LongTensor(np.where(labels)[1])
        adj = torch.FloatTensor(np.array(adj.todense()))

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        # data = (adj, features, labels)
        if self.train:
            data = features[idx_train]
            target = labels[idx_train]
        else:
            data = features[idx_test]
            target = labels[idx_test]

        if self.dataidxs is not None:
            data = features[self.dataidxs]
            target = labels[self.dataidxs]

        return data, adj, target

        # return , idx_train, idx_val, idx_test

        #
        # if self.train:
        #     data = mnist_dataobj.data
        #     target = np.array(mnist_dataobj.targets)
        # else:
        #     data = mnist_dataobj.data
        #     target = np.array(mnist_dataobj.targets)
        #
        # if self.dataidxs is not None:
        #     data = data[self.dataidxs]
        #     target = target[self.dataidxs]
        #
        # return data, target



    def __getitem__(self, index):

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
