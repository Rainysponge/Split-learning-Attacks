from .baseDatasetFactory import abstractDatasetFactory
from core.log.Log import Log
from .controller.cifar10Controller import cifar10Controller
from .controller.mnistController import mnistController
from .controller.adultController import adultController
from .controller.germanController import germanController
from .controller.fashionmnistController import fashionmnistController
from .controller.cheXpertController import cheXpertController
from .controller.coraController import coraController
from .controller.molhivController import molhivController
from .controller.shakespeareController import shakespeareController


class datasetFactory(abstractDatasetFactory):
    def __init__(self, parse):
        self.parse = parse
        self.log = Log(self.__class__.__name__, parse)

    def factory(self):
        # try:
        # self.log.info(sys.argv[0])
        self.log.info(self.parse["dataset"])
        if isinstance(self.parse["dataset"], list):
            # 特供方法 多任务
            dataset_cur, cur_client_num = self.judge_client_dataset()
            item = self.parse["dataset"][dataset_cur]
            create_controller = compile('{}Controller(self.{})'.format(item, "parse"),
                                        './core/dataset/{}/{}Controller.py'.format(item,
                                                                                   item),
                                        'eval')

            controller = eval(create_controller)

            return controller

        create_controller = compile('{}Controller(self.{})'.format(self.parse.dataset, "parse"),
                                    './core/dataset/{}/{}Controller.py'.format(self.parse.dataset, self.parse.dataset),
                                    'eval')
        # self.log.info('{}Controller({})'.format(self.parse.dataset, "parse"))
        # if self.parse["dataset"] == "fashionmnist":
        #     create_controller = compile('{}Controller(self.{})'.format("mnist", "parse"),
        #                                 './core/dataset/{}/{}Controller.py'.format("mnist",
        #                                                                            "mnist"),
        #                                 'eval')
        controller = eval(create_controller)

        return controller
        # except Exception as e:
        #     self.log.info(e)
        # self.log.info(dir(a))

    def judge_client_dataset(self):
        """
        对于多任务范式的特供方法，返回数据集索引和当前数据集有几个客户端使用
        """
        client_num = self.parse['client_split'][0]
        for i in range(len(self.parse['client_split'])):
            if self.parse['rank'] <= self.parse['client_split'][i]:
                return i, client_num
            client_num = self.parse['client_split'][i+1] - self.parse['client_split'][i]
