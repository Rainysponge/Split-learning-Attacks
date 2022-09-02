import numpy as np
import logging
import torch
from core.log.Log import Log

from .basePartition import abstractPartition


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    # log.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


class cifar10Partition(abstractPartition):
    def __init__(self, parse):
        self.parse = parse
        self.log = Log("cifar10Partition", parse)

    def partition_data(self):
        partition_method = self.parse['partition_method']
        if partition_method == 'homo':
            return self.homo
        elif partition_method == 'hetero':
            return self.hetero
        elif partition_method == 'base_on_class':
            return self.base_on_class
        elif partition_method == "base_on_attribute":
            return self.base_on_attribute
        elif partition_method == "vertical":
            return self.vertical
        elif partition_method == "base_on_class_intersection":
            return self.base_on_class_intersection

    def homo(self, load_data):
        if self.parse["variants_type"] == "TaskAgnostic":
            return self.task_agnostic_homo(load_data)
        self.log.info("homo")
        X_train, y_train, X_test, y_test = load_data()
        if isinstance(X_train, tuple):
            n_train = X_train[0].shape[0]
        elif isinstance(X_train, torch.utils.data.dataset.Subset):
            n_train = len(X_train)
        else:
            n_train = X_train.shape[0]

        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, self.parse['client_number'])
        net_dataidx_map = {i: batch_idxs[i] for i in range(self.parse['client_number'])}
        # traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
        traindata_cls_counts = None
        # log.info(net_dataidx_map)
        return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts

    def hetero(self, load_data):
        self.log.info("hetero")
        X_train, y_train, X_test, y_test = load_data()
        # class_num = len(np.unique(y_train))
        min_size = 0
        K = len(np.unique(y_train))
        N = y_train.shape[0]
        n_nets = self.parse["client_number"]
        alpha = self.parse["partition_alpha"]
        net_dataidx_map = {}
        idx_batch = [[] for _ in range(n_nets)]
        while min_size < 10:
            # idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                # self.log("".format())
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                # Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

        min0 = 100000000000000
        for i in range(n_nets):
            if len(net_dataidx_map[i]) < min0:
                min0 = len(net_dataidx_map[i])
        for i in range(n_nets):
            if len(net_dataidx_map[i]) > min0:
                np.random.shuffle(net_dataidx_map[i])
                net_dataidx_map[i] = net_dataidx_map[i][0: min0]

        return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts

    def base_on_class(self, load_data):
        n_nets = self.parse["client_number"]
        self.log.info("base_on_class")
        X_train, y_train, X_test, y_test = load_data()
        # self.log.info(type(X_train))
        K = len(np.unique(y_train))
        net_dataidx_map = {}
        if n_nets > K:
            mod = n_nets % K
            div = n_nets // K
            client_per_class = []
            client_idx = 0
            for i in range(K):
                num = div + 1 if mod > 0 else div
                client_per_class.append([i for i in range(client_idx, client_idx + num)])
                client_idx += num
                mod -= 1
            for k in range(K):
                client_list = client_per_class
                idxs = np.where(y_train == k)[0]
                np.random.shuffle(idxs)
                batch_idxs = np.array_split(idxs, len(client_list))
                net_dataidx_map = {i: batch_idxs[i] for i in range(len(client_list))}

                # idxs = np.random.permutation(total_num)
                # batch_idxs = np.array_split(idxs, self.parse['client_number'])
                # net_dataidx_map = {i: batch_idxs[i] for i in range(self.parse['client_number'])}
                # traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

            traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

        else:
            mod = K % n_nets
            class_idx = 0

            div = K // n_nets
            class_per_client = []
            net_dataidx_map = dict()
            for i in range(n_nets):
                num = div + 1 if mod > 0 else div
                class_per_client.append([i for i in range(class_idx, class_idx + num)])
                class_idx += num
                mod -= 1
            self.log.info("{}".format(class_per_client))
            for i in range(n_nets):
                batch_idxs = []

                for k in class_per_client[i]:
                    idx_k = np.where(y_train == k)[0]
                    batch_idxs.extend(list(idx_k))
                    # idxs = np.random.permutation(total_num)
                    # batch_idxs = np.array_split(idxs, self.parse['client_number'])
                net_dataidx_map[i] = batch_idxs
            min = 100000000000000
            for i in range(n_nets):
                if len(net_dataidx_map[i]) < min:
                    min = len(net_dataidx_map[i])
            for i in range(n_nets):
                if len(net_dataidx_map[i]) > min:
                    np.random.shuffle(net_dataidx_map[i])
                    net_dataidx_map[i] = net_dataidx_map[i][0: min]
            traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

        return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts

    def base_on_attribute(self, load_data):
        n_nets = self.parse["client_number"]
        self.log.info("base_on_class")
        X_train, y_train, X_test, y_test = load_data()
        attribute = int(self.parse["partition_method_attributes"])
        K = len(np.unique(X_train[:, attribute]))
        attribute_label = np.unique(X_train[:, attribute])
        self.log.info("attribute_label: {}".format(attribute_label))

        net_dataidx_map = {}
        if n_nets > K:
            mod = n_nets % K
            div = n_nets // K
            client_per_class = []
            client_idx = 0
            for i in range(K):
                num = div + 1 if mod > 0 else div
                client_per_class.append([i for i in range(client_idx, client_idx + num)])

                client_idx += num
                mod -= 1
            for k in range(0, -K, -1):
                client_list = client_per_class[-k]
                client_first = client_list[0]
                idxs = np.where(X_train[:, attribute] == k)[0]
                np.random.shuffle(idxs)
                batch_idxs = np.array_split(idxs, len(client_list))
                for i in client_list:
                    net_dataidx_map[i] = batch_idxs[i - client_first]

            traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

        else:
            mod = K % n_nets
            class_idx = 0

            div = K // n_nets
            class_per_client = []
            for i in range(n_nets):
                num = div + 1 if mod > 0 else div
                class_per_client.append([attribute_label[i] for i in range(class_idx, class_idx + num)])
                class_idx += num
                mod -= 1
            self.log.info("class_per_client : {}".format(class_per_client))

            for i in range(n_nets):
                batch_idxs = []

                for k in class_per_client[i]:
                    idx_k = np.where(X_train[:, attribute] == k)[0]
                    batch_idxs.extend(list(idx_k))
                    # idxs = np.random.permutation(total_num)
                    # batch_idxs = np.array_split(idxs, self.parse['client_number'])
                net_dataidx_map[i] = batch_idxs
            traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
            for k in net_dataidx_map.keys():
                self.log.info("{} : {}".format(k, net_dataidx_map[k]))
            # self.log.info("net_dataidx_map : {}".format(net_dataidx_map.keys()))
            min = 100000000000000
            for i in range(n_nets):
                if len(net_dataidx_map[i]) < min:
                    min = len(net_dataidx_map[i])
            for i in range(n_nets):
                if len(net_dataidx_map[i]) > min:
                    np.random.shuffle(net_dataidx_map[i])
                    net_dataidx_map[i] = net_dataidx_map[i][0: min]

        return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts

    def base_on_class_intersection(self, load_data):
        n_nets = self.parse["client_number"]
        class_number_per_client = self.parse['class_number_per_client']
        self.log.info("base_on_class_intersection")
        X_train, y_train, X_test, y_test = load_data()
        K = len(np.unique(y_train))

        if class_number_per_client * n_nets < K:
            return self.base_on_class(load_data)

        mod = K % n_nets
        class_idx = 0

        div = K // n_nets

        r = class_number_per_client - div


        class_per_client = []
        net_dataidx_map = dict()
        for i in range(n_nets):
            num = div + 1 if mod > 0 else div
            class_per_client.append([i % K for i in range(class_idx, class_idx + class_number_per_client)])
            class_idx += num
            mod -= 1
        self.log.info("{}".format(class_per_client))
        for i in range(n_nets):
            batch_idxs = []

            for k in class_per_client[i]:
                idx_k = np.where(y_train == k)[0]
                batch_idxs.extend(list(idx_k))
                # idxs = np.random.permutation(total_num)
                # batch_idxs = np.array_split(idxs, self.parse['client_number'])
            net_dataidx_map[i] = batch_idxs
        min = 100000000000000
        for i in range(n_nets):
            if len(net_dataidx_map[i]) < min:
                min = len(net_dataidx_map[i])
        for i in range(n_nets):
            if len(net_dataidx_map[i]) > min:
                np.random.shuffle(net_dataidx_map[i])
                net_dataidx_map[i] = net_dataidx_map[i][0: min]
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
        return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts




    def vertical(self, load_data):
        # n_nets = self.parse["client_number"]
        # self.log.info("base_on_class")
        # X_train, y_train, X_test, y_test = load_data()
        # self.log.info(type(X_train))
        # y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)
        #
        # X_train = X_train.split([7, 7], dim=1)
        # X_test = X_test.split([7, 7], dim=1)
        # # for i in range(self.parse["client_number"]):
        # #     net_dataidx_map
        # net_dataidx_train_map = {i: X_train[i] for i in range(n_nets)}
        # net_dataidx_test_map = {i: X_test[i] for i in range(n_nets)}
        # traindata_cls_counts = None
        # testdata_cls_counts = None
        #
        # return X_train, y_train, X_test, y_test, net_dataidx_train_map, traindata_cls_counts, \
        #        net_dataidx_test_map, testdata_cls_counts
        self.log.info("vertical")
        X_train, y_train, X_test, y_test = load_data()
        n_train = X_train.shape[0]
        net_dataidx_map = {i: [j for j in range(n_train)] for i in range(self.parse['client_number'])}

        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
        # log.info(net_dataidx_map)
        return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts

    def task_agnostic_homo(self, load_data):
        self.log.info("task_agnostic_homo")
        X_train, y_train, X_test, y_test = load_data()
        n_train = X_train.shape[0]
        total_num = n_train
        idxs = np.random.permutation(total_num)

        # 判别client属于的数据集 eg: parse['client_split'] = [[1,2,3],[4,5,6]] -> [3,6]
        dataset_cur, cur_client_num = self.judge_client_dataset()
        logging.info("dataset_cur: {}, cur_client_num: {}".format(dataset_cur, cur_client_num))

        batch_idxs = np.array_split(idxs, cur_client_num)
        # net_dataidx_map = {i: batch_idxs[i] for i in range(self.parse['client_number'])}
        # net_dataidx_map = dict()
        if dataset_cur == 0:
            net_dataidx_map = {i: batch_idxs[i] for i in range(cur_client_num)}
        else:
            pre = self.parse['client_split'][dataset_cur-1]
            net_dataidx_map = {pre + i: batch_idxs[i] for i in range(cur_client_num)}

        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
        # log.info(net_dataidx_map)
        return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts

    def judge_client_dataset(self):
        """
        对于多任务范式的特供方法，返回数据集索引和当前数据集有几个客户端使用
        """
        client_num = self.parse['client_split'][0]
        for i in range(len(self.parse['client_split'])):
            if self.parse['rank'] <= self.parse['client_split'][i]:
                return i, client_num
            client_num = self.parse['client_split'][i+1] - self.parse['client_split'][i]
