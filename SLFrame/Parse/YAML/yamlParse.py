import yaml

from ..abstractParse import parse
from ..utlis import setParseAttribute
from core.log.Log import Log


class yamlParse(parse):
    """
    样例
    """
    def __init__(self):
        self.log = Log("yamlParse")
        self.dataset = ""
        self.model = ""
        self.dataDir = ""
        self.partition_method = ""
        self.partition_alpha = 0.5
        self.client_number = 16
        self.batch_size = 64
        self.lr = 0.001
        self.wd = 0.001
        self.epochs = 1
        self.comm_round = 10
        self.frequency_of_the_test = 1
        self.gpu_server_num = 1
        self.gpu_num_per_server = 4

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
