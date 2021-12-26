import abc


class abstractPartition(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def partition_data(self):
        pass

