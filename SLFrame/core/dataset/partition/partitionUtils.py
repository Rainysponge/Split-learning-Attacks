"""
用于分割数据集
"""
import numpy as np
from core.log.Log import Log

log = Log("partition.py")


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    log.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


class partition_method_list:
    """
    成员方法全是静态方法
    """

    @staticmethod
    def homo(parse, load_data):
        log.info("homo")
        X_train, y_train, X_test, y_test = load_data()
        n_train = X_train.shape[0]
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, parse['client_number'])
        net_dataidx_map = {i: batch_idxs[i] for i in range(parse['client_number'])}
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
        log.info(traindata_cls_counts)
        return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts

    @staticmethod
    def hetero_fix(parse, load_data):
        log.info("hetero-fix")
