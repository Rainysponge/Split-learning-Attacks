import logging
import os
import json

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import MNIST

from core.log.Log import Log
import h5py
import torch.utils.data as data

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


class shakespeare_truncated(data.Dataset):

    def __init__(self, parse, dataidxs=None, train=True, transform=None, client_idx=None):

        self.parse = parse
        self.target_transform = None
        self.transform = transform
        self.data = None
        self.dataidxs = dataidxs
        self.log = Log(self.__class__.__name__, parse)
        self.root = parse['dataDir'] + "shakespeare/"
        self.target = None
        self.bantch_size = parse["batch_size"]
        # self.n_nets = parse['client_number'] if parse['client_number'] else 16
        self.train = train

        self.download = parse['download'] if parse['download'] is not None else False

        self.data, self.target = self.__build_truncated_dataset__(client_idx)

    def __build_truncated_dataset__(self, client_idx=None):
        train_h5 = h5py.File(os.path.join(self.root, DEFAULT_TRAIN_FILE), 'r')
        test_h5 = h5py.File(os.path.join(self.root, DEFAULT_TEST_FILE), 'r')
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
        train_h5.close()
        test_h5.close()
        if self.train:
            return train_x, train_y
        else:
            return test_x, test_y
        # return train_x, train_y, test_x, test_y

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

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
