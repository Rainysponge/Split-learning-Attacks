import abc


class abstractDatasetFactory(metaclass=abc.ABCMeta):
    """
    用于按照配置信息生成相应数据集管理类的抽象工厂
    """
    @abc.abstractmethod
    def factory(self, datasetName):
        pass
