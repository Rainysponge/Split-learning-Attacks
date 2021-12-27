import yaml

from ..baseParse import parse
from ..utlis import setParseAttribute
from core.log.Log import Log


class yamlParse(parse):
    """
    样例
    """

    def __init__(self, **kwargs):
        self.log = Log("yamlParse")
        self.dataset = ""
        self.model = ""
        self.dataDir = ""
        self.partition_method = ""
        self.partition_alpha = 0.5
        self.split_layer = 1
        self.client_number = 16
        self.batch_size = 64
        self.lr = 0.001
        self.wd = 0.001
        self.epochs = 3
        self.comm_round = 10
        self.log_step = 50
        self.frequency_of_the_test = 1
        self.gpu_server_num = 1
        self.gpu_num_per_server = 4
        self.kwargs = kwargs

    def save(self, data, filePath, *args, **kwargs):
        """
        将dict变为yaml文件输出，如果path下该文件则覆盖
        """
        self.log.info("save to path: {}".format(filePath))
        try:
            with open(filePath, "w") as f:
                yaml.dump(data, f, **kwargs)
        except Exception as e:
            self.log.error(e)

    def load(self, filePath, *args, **kwargs):
        try:
            with open(filePath, 'r') as f:
                d = yaml.load(f, Loader=yaml.FullLoader)
            setParseAttribute(self, d)
            self.log.info("load config successfully!")
            return d
        except Exception as e:
            self.log.error(e)

    def __getitem__(self, item):
        """
        可以通过成员变量访问变量，也可以用字典访问，默认返回None，可以用于拓展
        """
        # return vars(self)[item] if vars(self)[item] else self.kwargs[item]
        if item in vars(self):
            return vars(self)[item]
        if item in self.kwargs:
            return self.kwargs[item]

        return None

    def __setitem__(self, key, value):
        if key in vars(self):
            setattr(self, key, value)
        else:
            self.kwargs[key] = value

