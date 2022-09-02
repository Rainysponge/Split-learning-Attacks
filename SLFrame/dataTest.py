from core.dataset.controller.cheXpertController import cheXpertController
from core.log.Log import Log
from core.dataset.datasetFactory import datasetFactory
from Parse.parseFactory import parseFactory, JSON, YAML
from core.splitApi import SplitNN_distributed, SplitNN_init
from core.dataset.dataset.cheXpert import cheXpert_truncated
import torch.utils.data as data
from core.dataset.controller.cifar10Controller import cifar10Controller
from core.dataset.controller.mnistController import mnistController
from core.dataset.controller.adultController import adultController
from core.dataset.controller.germanController import germanController
from core.dataset.controller.fashionmnistController import fashionmnistController
from core.dataset.controller.cheXpertController import cheXpertController

if __name__ == '__main__':
    args = parseFactory(fileType=YAML).factory()
    args.load('./config.yaml')
    # # comm, process_id, worker_number = SplitNN_init(parse=args)
    # # comm, process_id, worker_number = SplitNN_init(args)
    # # c = cheXpertController(args)
    # train_ds = cheXpert_truncated(parse=args, transform=None, dataidxs=None)
    # test_ds = cheXpert_truncated(parse=args, transform=None, train=False)
    #
    # train_dl = data.DataLoader(dataset=train_ds, batch_size=1, shuffle=True, drop_last=True)
    # test_dl = data.DataLoader(dataset=test_ds, batch_size=1, shuffle=False, drop_last=True)
    #
    # dataloader = iter(train_dl)
    controller_list = []
    for item in ['mnist', 'fashionmnist']:
        create_controller = compile('{}Controller({})'.format(item, "args"),
                                    './core/dataset/{}/{}Controller.py'.format(item,
                                                                               item),
                                    'eval')

        controller = eval(create_controller)
        controller_list.append(controller)
    print(controller_list)
