import numpy as np
import re
import torch
from torch.utils.data.dataset import T_co
import torchvision.transforms as transforms

import torch.utils.data as data

from ..dataset.shakespeare import shakespeare_truncated
from ..partition.partitionFactory import partitionFactory
from core.log.Log import Log

import logging
import os
import json

import torch

from core.log.Log import Log
import h5py

# import utils
import collections

word_dict = None
word_list = None
_pad = '<pad>'
_bos = '<bos>'
_eos = '<eos>'
'''
This code follows the steps of preprocessing in tff shakespeare dataset: 
https://github.com/google-research/federated/blob/master/utils/datasets/shakespeare_dataset.py
'''

SEQUENCE_LENGTH = 80  # from McMahan et al AISTATS 2017
# Vocabulary re-used from the Federated Learning for Text Generation tutorial.
# https://www.tensorflow.org/federated/tutorials/federated_learning_for_text_generation
CHAR_VOCAB = list(
    'dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r'
)


def get_word_dict():
    global word_dict
    if word_dict == None:
        words = [_pad] + CHAR_VOCAB + [_bos] + [_eos]
        word_dict = collections.OrderedDict()
        for i, w in enumerate(words):
            word_dict[w] = i
    return word_dict


def get_word_list():
    global word_list
    if word_list == None:
        word_dict = get_word_dict()
        word_list = list(word_dict.keys())
    return word_list


def id_to_word(idx):
    return get_word_list()[idx]


def char_to_id(char):
    word_dict = get_word_dict()
    if char in word_dict:
        return word_dict[char]
    else:
        return len(word_dict)


def preprocess(sentences, max_seq_len=SEQUENCE_LENGTH):
    sequences = []

    def to_ids(sentence, num_oov_buckets=1):
        '''
        map list of sentence to list of [idx..] and pad to max_seq_len + 1
        Args:
            num_oov_buckets : The number of out of vocabulary buckets.
            max_seq_len: Integer determining shape of padded batches.
        '''
        tokens = [char_to_id(c) for c in sentence]
        tokens = [char_to_id(_bos)] + tokens + [char_to_id(_eos)]
        if len(tokens) % (max_seq_len + 1) != 0:
            pad_length = (-len(tokens)) % (max_seq_len + 1)
            tokens += [char_to_id(_pad)] * pad_length
        return (tokens[i:i + max_seq_len + 1]
                for i in range(0, len(tokens), max_seq_len + 1))

    for sen in sentences:
        sequences.extend(to_ids(sen))
    return sequences


def split(dataset):
    ds = np.asarray(dataset)
    x = ds[:, :-1]
    y = ds[:, 1:]
    return x, y


client_ids_train = None
client_ids_test = None
DEFAULT_TRAIN_CLIENTS_NUM = 715
DEFAULT_TEST_CLIENTS_NUM = 715
DEFAULT_BATCH_SIZE = 4
DEFAULT_TRAIN_FILE = 'shakespeare_train.h5'
DEFAULT_TEST_FILE = 'shakespeare_test.h5'

# group name defined by tff in h5 file
_EXAMPLE = 'examples'
_SNIPPETS = 'snippets'


class shakespeareController():
    def __init__(self, parse, transform=None):
        self.parse = parse
        self.target_transform = None
        self.transform = transform
        self.data = None
        self.log = Log(self.__class__.__name__, parse)
        self.root = parse['dataDir']
        self.target = None
        self.batch_size = parse["batch_size"]
        # self.n_nets = parse['client_number'] if parse['client_number'] else 16
        self.train = parse['train'] if parse['train'] is not None else True
        self.download = parse['download'] if parse['download'] is not None else False

    def loadData(self, client_idx=None):
        self.log.info(self.parse.dataDir)
        # train_transform, test_transform = _data_transforms_mnist()
        # shakespeare_train_ds = shakespeare_truncated(parse=self.parse)
        # shakespeare_test_ds = shakespeare_truncated(parse=self.parse, train=False)
        # X_train, y_train = shakespeare_train_ds.data, shakespeare_train_ds.target
        # X_test, y_test = shakespeare_test_ds.data, shakespeare_test_ds.target
        # train_ds = data.TensorDataset(torch.tensor(X_train[:, :]),
        #                               torch.tensor(y_train[:]))
        # test_ds = data.TensorDataset(torch.tensor(X_test[:, :]),
        #                              torch.tensor(y_test[:]))
        # train_dl = data.DataLoader(dataset=train_ds,
        #                            batch_size=self.batch_size,
        #                            shuffle=True,
        #                            drop_last=False)
        # test_dl = data.DataLoader(dataset=test_ds,
        #                           batch_size=self.batch_size,
        #                           shuffle=True,
        #                           drop_last=False)
        #
        # return train_dl, test_dl
        train_h5 = h5py.File(os.path.join(self.root + "shakespeare/", DEFAULT_TRAIN_FILE), 'r')
        test_h5 = h5py.File(os.path.join(self.root + "shakespeare/", DEFAULT_TEST_FILE), 'r')
        train_ds = []
        test_ds = []

        # load data
        if client_idx is None:
            # get ids of all clients
            train_ids = client_ids_train
            test_ids = client_ids_test
        else:
            # get ids of single client
            train_ids = [client_ids_train[client_idx]]
            test_ids = [client_ids_test[client_idx]]

        for client_id in train_ids:
            raw_train = train_h5[_EXAMPLE][client_id][_SNIPPETS][()]
            raw_train = [x.decode('utf8') for x in raw_train]
            train_ds.extend(preprocess(raw_train))
        for client_id in test_ids:
            raw_test = test_h5[_EXAMPLE][client_id][_SNIPPETS][()]
            raw_test = [x.decode('utf8') for x in raw_test]
            test_ds.extend(preprocess(raw_test))

        # split data
        train_x, train_y = split(train_ds)
        test_x, test_y = split(test_ds)
        train_ds = data.TensorDataset(torch.tensor(train_x[:, :]),
                                      torch.tensor(train_y[:]))
        test_ds = data.TensorDataset(torch.tensor(test_x[:, :]),
                                     torch.tensor(test_y[:]))
        train_dl = data.DataLoader(dataset=train_ds,
                                   batch_size=self.batch_size,
                                   shuffle=True,
                                   drop_last=False)
        test_dl = data.DataLoader(dataset=test_ds,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  drop_last=False)

        train_h5.close()
        test_h5.close()
        return train_dl, test_dl

    # def partition_data(self):
    #     partition_method = partitionFactory(parse=self.parse).factory()
    #
    #     self.log.info(partition_method)
    #     return partition_method(self.loadData)

    # def get_dataloader(self, dataidxs=None):
    #
    #     dl_obj = mnist_truncated
    #
    #     transform_train, transform_test = _data_transforms_mnist()
    #
    #     train_ds = dl_obj(parse=self.parse, transform=transform_train, dataidxs=dataidxs)
    #     test_ds = dl_obj(parse=self.parse, transform=transform_test, train=False)
    #
    #     train_dl = data.DataLoader(dataset=train_ds, batch_size=self.bantch_size, shuffle=True, drop_last=True)
    #     test_dl = data.DataLoader(dataset=test_ds, batch_size=self.bantch_size, shuffle=False, drop_last=True)
    #
    #     return train_dl, test_dl

    def load_partition_data(self, process_id):
        # X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = self.partition_data()
        # class_num = len(np.unique(y_train))
        # # self.log.info("class_num = " + str(class_num))
        # train_data_num = sum([len(net_dataidx_map[r]) for r in net_dataidx_map.keys()])
        #
        # # get global test data
        # if process_id == 0:
        #     train_data_global, test_data_global = self.get_dataloader()
        #     self.log.info("train_dl_global number = " + str(len(train_data_global)))
        #     self.log.info("test_dl_global number = " + str(len(test_data_global)))
        #     train_data_local = None
        #     test_data_local = None
        #     local_data_num = 0
        # else:
        #     # get local dataset
        #     dataidxs = net_dataidx_map[process_id - 1]
        #     local_data_num = len(dataidxs)
        #     self.log.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
        #     # training batch size = 64;
        #     train_data_local, test_data_local = self.get_dataloader(dataidxs)
        #     # self.log.info("dataidxs: {}".format(dataidxs))
        #     self.log.info("process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
        #         process_id, len(train_data_local), len(test_data_local)))
        #     train_data_global = None
        #     test_data_global = None
        # self.parse["trainloader"] = train_data_local
        # self.parse["testloader"] = test_data_local
        # self.parse["train_data_num"] = train_data_num
        # self.parse["train_data_global"] = train_data_global
        # self.parse["test_data_global"] = test_data_global
        # self.parse["local_data_num"] = local_data_num
        # self.parse["class_num"] = class_num
        # return train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num
        if process_id == 0:
            # get global dataset
            train_data_global, test_data_global = self.loadData(process_id - 1)
            train_data_num = len(train_data_global)
            test_data_num = len(test_data_global)
            logging.info("train_dl_global number = " + str(train_data_num))
            logging.info("test_dl_global number = " + str(test_data_num))
            train_data_local = None
            test_data_local = None
            local_data_num = 0
        else:
            # get local dataset
            # client id list
            train_file_path = os.path.join(self.root + "shakespeare/", DEFAULT_TRAIN_FILE)
            test_file_path = os.path.join(self.root + "shakespeare/", DEFAULT_TEST_FILE)
            with h5py.File(train_file_path, 'r') as train_h5, h5py.File(test_file_path, 'r') as test_h5:
                global client_ids_train, client_ids_test
                client_ids_train = list(train_h5[_EXAMPLE].keys())
                client_ids_test = list(test_h5[_EXAMPLE].keys())

            train_data_local, test_data_local = self.loadData(process_id - 1)
            train_data_num = local_data_num = len(train_data_local.dataset)
            logging.info("rank = %d, local_sample_number = %d" %
                         (process_id, local_data_num))
            train_data_global = None
            test_data_global = None

        VOCAB_LEN = len(get_word_dict()) + 1
        self.parse["trainloader"] = train_data_local
        self.parse["testloader"] = test_data_local
        self.parse["train_data_num"] = train_data_num
        self.parse["train_data_global"] = train_data_global
        self.parse["test_data_global"] = test_data_global
        self.parse["local_data_num"] = local_data_num
        self.parse["class_num"] = VOCAB_LEN
        return train_data_num, train_data_global, test_data_global, \
               local_data_num, train_data_local, test_data_local, VOCAB_LEN
        # return client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        #        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, output_dim

        # return train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, \
        #        test_data_local, class_num

# train_ds = data.TensorDataset(torch.tensor(train_x[:, :]),
#                                   torch.tensor(train_y[:]))
#     test_ds = data.TensorDataset(torch.tensor(test_x[:, :]),
#                                  torch.tensor(test_y[:]))
#     train_dl = data.DataLoader(dataset=train_ds,
#                                batch_size=train_bs,
#                                shuffle=True,
#                                drop_last=False)
#     test_dl = data.DataLoader(dataset=test_ds,
#                               batch_size=test_bs,
#                               shuffle=True,
#                               drop_last=False)
