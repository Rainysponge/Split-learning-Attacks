from .baseDatasetFactory import abstractDatasetFactory
from core.log.Log import Log
from .controller.cifar10Controller import cifar10Controller
from .controller.mnistController import mnistController
from .controller.adultController import adultController
from .controller.germanController import germanController


class datasetFactory(abstractDatasetFactory):
    def __init__(self, parse):
        self.parse = parse
        self.log = Log(self.__class__.__name__, parse)

    def factory(self):
        # try:
        # self.log.info(sys.argv[0])
        self.log.info(self.parse["dataset"])

        create_controller = compile('{}Controller(self.{})'.format(self.parse.dataset, "parse"),
                                    './core/dataset/{}/{}Controller.py'.format(self.parse.dataset, self.parse.dataset),
                                    'eval')
        # self.log.info('{}Controller({})'.format(self.parse.dataset, "parse"))
        controller = eval(create_controller)
        return controller
        # except Exception as e:
        #     self.log.info(e)
        # self.log.info(dir(a))
