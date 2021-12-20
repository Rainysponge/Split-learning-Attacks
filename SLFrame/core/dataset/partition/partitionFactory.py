import re
from core.log.Log import Log

from .basePartitionFactory import abstractPartitionFactory
from .partitionUtils import partition_method_list


class partitionFactory(abstractPartitionFactory):
    def __init__(self, parse):
        self.log = Log(self.__class__.__name__)
        self.parse = parse

    def factory(self):
        partition_method = self.parse.partition_method
        self.log.info("partition_method is {}".format(partition_method))

        # 动态调用 后面把partition_method_list的路径放到配置文件里面，拓展性上继承partition_method_list，然后重载或添加方法
        method_list = [str for str in dir(partition_method_list) if re.match("__*", str) is None]
        if partition_method in method_list:
            return getattr(partition_method_list, partition_method)

        # if partition_method == "homo":
        #     return homo
        # if partition_method == "hetero-fix":
        #     return hetero_fix
        raise NameError("NameError: partition_method: {}未找到".format(partition_method))
