import torch
import logging
import torch.nn as nn

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim

# import setproctitle
import sys

sys.path.extend("../")
sys.path.extend("../../")

from core.log.Log import Log
from core.dataset.datasetFactory import datasetFactory
from Parse.parseFactory import parseFactory, JSON, YAML
from core.model.cnn import CNN_OriginalFedAvg, Net, cifar10_client, cifar10_server
from core.model.models import LeNetComplete, LeNetClientNetwork, LeNetServerNetwork, adult_LR_client, adult_LR_server, \
    german_LR_client, german_LR_server
from core.model.resnet import resnet56, ResNet_client, ResNet_server
from core.variants.vanilla.client import SplitNNClient
from core.variants.vanilla.server import SplitNNServer
from core.splitApi import SplitNN_distributed, SplitNN_init

# log = Log("Test.py")

client_model = german_LR_client()
server_model = german_LR_server()


def init_training_device(process_ID, fl_worker_num, gpu_num_per_machine):
    # initialize the mapping from process ID to GPU ID: <process ID, GPU ID>
    # logging = Logging("init_training_device")
    if process_ID == 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device
    process_gpu_dict = dict()
    for client_index in range(fl_worker_num):
        gpu_index = client_index % gpu_num_per_machine
        process_gpu_dict[client_index] = gpu_index

    logging.info(process_gpu_dict)
    device = torch.device("cuda:" + str(process_gpu_dict[process_ID - 1]) if torch.cuda.is_available() else "cpu")
    logging.info(device)
    return device


if __name__ == '__main__':
    """
        尽量让Test.py是可以不需要做任何其他操作直接运行的
    """
    args = parseFactory(fileType=YAML).factory()
    """
           解析器展示, 把需要的数据放在d里面，或者在下面的args里面 eg：args["model"] = client_model
           这里写过一次之后就可以注释掉了，存在文件中了
       """
    # d = {
    #     "dataset": "cifar10",
    #     "dataDir": "./data/cifar10",
    #     'download': True,
    #     'partition_method': 'homo',
    #     'log_step': 20,
    #     "rank": 1,
    #     "max_rank": 5,
    #     "lr": 0.01,
    #     "server_rank": 0,
    #     'device': 'cpu',
    #
    # }
    #
    # args.save(d, './config.yaml')
    d = args.load('./config.yaml')
    # comm, process_id, worker_number = SplitNN_init(parse=args)
    comm, process_id, worker_number = SplitNN_init(args)
    args["rank"] = process_id  # 设置当前process_id

    args["client_model"] = client_model
    args["server_model"] = server_model
    device = init_training_device(process_id, worker_number - 1, args.gpu_num_per_server)
    args["device"] = device
    log = Log("Test.py", args)
    # log.info(device)

    dataset = datasetFactory(args).factory()  # loader data and partition method
    # print(dataset)

    train_data_num, train_data_global, test_data_global, local_data_num, \
    train_data_local, test_data_local, class_num = dataset.load_partition_data(process_id)  # 这里的4是process Id


    server = SplitNNServer(args)
    SplitNN_distributed(process_id, args)
