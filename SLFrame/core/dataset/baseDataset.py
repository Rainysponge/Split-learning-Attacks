import abc
import torch.utils.data as data


class dataset(data.Dataset, metaclass=abc.ABCMeta):
    """
    管理数据集的抽象类
    datasetFactory 中 import datasetController
    datasetController 中 import partitionFactory
    partitionFactory 中 import partition_method_list
    """
    # def __init__(self, dataDir):
    #     self.dataDir = dataDir

    @abc.abstractmethod
    def loadData(self):
        pass

    @abc.abstractmethod
    def partition_data(self):
        pass
