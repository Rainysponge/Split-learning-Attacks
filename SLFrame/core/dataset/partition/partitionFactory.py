import re
from core.log.Log import Log

from .basePartitionFactory import abstractPartitionFactory
from .cifar10Partition import cifar10Partition


class partitionFactory(abstractPartitionFactory):
    def __init__(self, parse):
        self.log = Log(self.__class__.__name__, parse)
        self.parse = parse

    def factory(self):
        dataset = self.parse['dataset']
        self.log.info("dataset is {}".format(dataset))

        # # 动态调用 后面把partition_method_list的路径放到配置文件里面，拓展性上继承partition_method_list，然后重载或添加方法
        # method_list = [str for str in dir(partition_method_list) if re.match("__*", str) is None]
        # if partition_method in method_list:
        #     return getattr(partition_method_list, partition_method)

        if dataset == 'cifar10' or dataset == "mnist" or dataset == "adult" or dataset == 'german' \
                or dataset == "fashionmnist":
            partition = cifar10Partition(self.parse).partition_data()
            return partition
        else:
            self.log.error("dataset {} can't be found".format(dataset))

        # if partition_method == "homo":
        #     return homo
        # if partition_method == "hetero-fix":
        #     return hetero_fix
        raise NameError("NameError: dataset: {}未找到".format(dataset))
