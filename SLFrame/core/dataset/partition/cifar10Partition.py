import numpy as np
from core.log.Log import Log

from .basePartition import abstractPartition

log = Log("cifar10Partition")


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    log.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


class cifar10Partition(abstractPartition):
    def __init__(self, parse):
        self.parse = parse

    def partition_data(self):
        partition_method = self.parse['partition_method']
        if partition_method == 'homo':
            return self.homo

    def homo(self, load_data):
        log.info("homo")
        X_train, y_train, X_test, y_test = load_data()
        n_train = X_train.shape[0]
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, self.parse['client_number'])
        net_dataidx_map = {i: batch_idxs[i] for i in range(self.parse['client_number'])}
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
        log.info(net_dataidx_map)
        return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts
