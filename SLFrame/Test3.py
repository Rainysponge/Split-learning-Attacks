import numpy as np
import pandas as pd
import torch.nn as nn

from matplotlib import pyplot as plt
from torchvision.datasets import MNIST
from core.model.models import german_LR_client, german_LR_server, LeNetClientNetwork,  LeNetServerNetwork,\
    adult_LR_client, adult_LR_server, LeNetComplete


class A():
    def __init__(self, t):
        self.a = 0
        self.b = None
        self.f(t)

    def f(self, t):
        if t == 1:
            self.b = 1
        else:
            self.b = 9


if __name__ == "__main__":
    # mnist_dataobj = MNIST('./data/mnist')
    # data = mnist_dataobj.data
    # y_train = np.array(mnist_dataobj.targets)
    #
    # min_size = 0
    # K = len(np.unique(y_train))
    # N = y_train.shape[0]
    # n_nets = 3
    # alpha = 100.0
    # net_dataidx_map = {}
    # idx_batch = [[] for _ in range(n_nets)]
    # while min_size < 10:
    #     # idx_batch = [[] for _ in range(n_nets)]
    #     # for each class in the dataset
    #     for k in range(K):
    #         idx_k = np.where(y_train == k)[0]
    #         # self.log("".format())
    #         np.random.shuffle(idx_k)
    #         proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
    #         # Balance
    #         proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
    #         proportions = proportions / proportions.sum()
    #         proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
    #         idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
    #         min_size = min([len(idx_j) for idx_j in idx_batch])
    #
    # for j in range(n_nets):
    #     np.random.shuffle(idx_batch[j])
    #     net_dataidx_map[j] = idx_batch[j]
    # # traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    #
    # # print(y_train[11491])
    # fig = plt.figure()
    # ax_dict = dict()
    # for k in net_dataidx_map.keys():
    #     ax_dict[k] = fig.add_subplot(131+k)
    #     # print(len(net_dataidx_map[k]))
    #     print(k)
    #     sum_dict = dict()
    #     for i in range(10):
    #         sum_dict[i] = 0
    #     for i in net_dataidx_map[k]:
    #         sum_dict[y_train[i]] = sum_dict[y_train[i]] + 1
    #     print(sum_dict)
    #     df = pd.DataFrame.from_dict(sum_dict, orient='index', columns=['number'])
    #     df = df.reset_index().rename(columns={"index": 'label'})
    #     sns.barplot(x='label', y='number', data=df, ax=ax_dict[k])
    # plt.title("α = {}".format(alpha))
    # plt.show()

    # i = 0
    # a = A(12)
    # print(a.b)
    model = LeNetComplete()
    # # cm2 = LeNetClientNetworkPart2()
    # fc_features = model.fc.in_features
    # model.fc = nn.Sequential(nn.Flatten(), nn.Linear(fc_features, 10))
    client_model = nn.Sequential(*nn.ModuleList(model.children())[:2])
    server_model = nn.Sequential(*nn.ModuleList(model.children())[2:])
    print(client_model)
    print(server_model)
