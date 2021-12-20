"""
对应json， yaml解析器的基类
"""
import abc

JSON = 1
YAML = 2


class parse(metaclass=abc.ABCMeta):
    """
    解析器至少应该有save和load
    """

    @abc.abstractmethod
    def save(self, data, filePath, *args, **kwargs):
        """
        将字典存入文件中
        """
        pass

    @abc.abstractmethod
    def load(self, filePath, *args, **kwargs):
        """
        读取文件中的信息，存入parse的对应属性中，返回一个字典
        """
        pass


class abstractParseFactory(metaclass=abc.ABCMeta):
    """
    用于用户继承来动态修改其中解析器配置
    """
    @abc.abstractmethod
    def factory(self):
        """
        根据所给参数，生成相应解析器
        """
        pass



