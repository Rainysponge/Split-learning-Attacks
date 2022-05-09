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
from core.model.models_for_U import LeNetClientNetworkPart1, LeNetClientNetworkPart2, \
    adult_LR_U_client1, adult_LR_U_client2, adult_LR_U_server
from core.model.models import german_LR_client, german_LR_server, LeNetClientNetwork, LeNetServerNetwork, \
    adult_LR_client, adult_LR_server
from core.model.Alex_model import AlexNet_client, AlexNet_server
from core.model.resnet import resnet56, ResNet_client, ResNet_server
from core.model.EffientNet0 import EfficientNetB0_Client, EfficientNetB0_Server

from core.variants.vanilla.client import SplitNNClient
from core.variants.vanilla.server import SplitNNServer
from core.splitApi import SplitNN_distributed, SplitNN_init


# client_model = [adult_LR_U_client1(), adult_LR_U_client2()]
client_model = AlexNet_client()
server_model = AlexNet_server()


def init_training_device(process_ID, fl_worker_num, gpu_num_per_machine,dv):
    # initialize the mapping from process ID to GPU ID: <process ID, GPU ID>
    # logging = Logging("init_training_device")
    if(dv=="cpu"):
        return torch.device("cpu")

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
    args.load('./config.yaml')
    # comm, process_id, worker_number = SplitNN_init(parse=args)
    comm, process_id, worker_number = SplitNN_init(args)
    args["rank"] = process_id  # 设置当前process_id

    args["client_model"] = client_model
    args["server_model"] = server_model
    device = init_training_device(process_id, worker_number - 1, args.gpu_num_per_server,args["device"])
    args["device"] = device

    dataset = datasetFactory(args).factory()  # loader data and partition method

    train_data_num, train_data_global, test_data_global, local_data_num, \
     train_data_local, test_data_local, class_num = dataset.load_partition_data(process_id)  # 这里的4是process Id

    log = Log("main", args)
    # log.info("{}".format(train_data_num))

    # str_process_name = "SplitNN (distributed):" + str(process_id)
    # setproctitle.setproctitle(str_process_name
    SplitNN_distributed(process_id, args)
