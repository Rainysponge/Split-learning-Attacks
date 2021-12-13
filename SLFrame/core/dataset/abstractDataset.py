import abc
import torch.utils.data as data


class dataset(data.Dataset, metaclass=abc.ABCMeta):
    """
    管理数据集的抽象类
    """
    # def __init__(self, dataDir):
    #     self.dataDir = dataDir

    @abc.abstractmethod
    def loadData(self):
        pass

    @abc.abstractmethod
    def partition_data(self, dataset, partition, n_nets, alpha):
        pass
