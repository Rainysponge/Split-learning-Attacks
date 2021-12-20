import abc


class abstractPartition(metaclass=abc.ABCMeta):
    @abc.ABCMeta
    def partition_data(self):
        pass

    @abc.ABCMeta
    def __build_truncated_dataset__(self):
        pass
