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

