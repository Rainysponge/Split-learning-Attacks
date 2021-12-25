import abc


class abstractPartitionFactory(metaclass=abc.ABCMeta):
    """
    按照配置文件生成相应的分割方法
    """
    @abc.abstractmethod
    def factory(self):
        pass
