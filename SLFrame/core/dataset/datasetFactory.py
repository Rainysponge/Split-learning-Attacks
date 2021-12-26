from .baseDatasetFactory import abstractDatasetFactory
from core.log.Log import Log
from .controller.cifar10Controller import cifar10Controller
from .controller.mnistController import mnistController


class datasetFactory(abstractDatasetFactory):
    def __init__(self):
        self.log = Log(self.__class__.__name__)

    def factory(self, parse):
        # try:
        # self.log.info(sys.argv[0])
        self.log.info(parse.dataset)
        create_controller = compile('{}Controller({})'.format(parse.dataset, "parse"),
                                    './core/dataset/{}/{}Controller.py'.format(parse.dataset, parse.dataset),
                                    'eval')
        controller = eval(create_controller)
        return controller
        # except Exception as e:
        #     self.log.info(e)
        # self.log.info(dir(a))
