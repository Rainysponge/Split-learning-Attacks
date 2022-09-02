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
from core.model.model_factory import model_factory
from core.model.GNN import GCN_server, GCN_client
from core.model.cnn import CNN_OriginalFedAvg, Net, cifar10_client, cifar10_server
from core.model.models_for_U import LeNetClientNetworkPart1, LeNetClientNetworkPart2, LeNetServerNetwork_U,\
    adult_LR_U_client1, adult_LR_U_client2, adult_LR_U_server
from core.model.models import german_LR_client, german_LR_server, LeNetClientNetwork,  LeNetServerNetwork,\
    adult_LR_client, adult_LR_server, LeNetComplete
from core.model.Alex_model import AlexNet_client, AlexNet_server
from core.model.resnet import resnet56, ResNet_client, ResNet_server
from core.model.EffientNet0 import EfficientNetB0_Client, EfficientNetB0_Server
from core.model.GNN import DMPNNEncoder, DMPNNEncoder_head


from core.variants.vanilla.client import SplitNNClient
from core.variants.vanilla.server import SplitNNServer
from core.splitApi import SplitNN_distributed, SplitNN_init

num_epochs = 3
hidden_size = 300
depth = 3
out_dim = 1

# head = nn.Sequential(
#     nn.Linear(hidden_size, hidden_size, bias=True),
#     nn.ReLU(),
#     nn.Linear(hidden_size, out_dim, bias=True),
# )
# model = nn.Sequential(
#     DMPNNEncoder(
#         hidden_size,
#         dataset.num_node_features,
#         dataset.num_edge_features,
#         depth,
#     ),
#     head,
# )
# client_model = [LeNetClientNetworkPart1(), LeNetClientNetworkPart2()]
client_model = DMPNNEncoder(
        hidden_size,
        9,
        3,
        depth,
)
server_model = DMPNNEncoder_head(300, out_dim, 9, 3, 3, True)
# client_model = LeNetClientNetwork()
# server_model = LeNetServerNetwork()
# model = LeNetComplete()
# # # cm2 = LeNetClientNetworkPart2()
# # fc_features = model.fc.in_features
# # model.fc = nn.Sequential(nn.Flatten(), nn.Linear(fc_features, 10))
# client_model = nn.Sequential(*nn.ModuleList(model.children())[:2])
# server_model = nn.Sequential(*nn.ModuleList(model.children())[2:])
# self.model = args["client_model"]
# self.model_2 = args["client_model_2"]


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
    args.load('./config.yaml')
    # comm, process_id, worker_number = SplitNN_init(parse=args)
    comm, process_id, worker_number = SplitNN_init(args)
    args["rank"] = process_id  # 设置当前process_id

    # args["client_model"], args["server_model"] = model_factory(args).create()
    args["client_model"], args["server_model"] = client_model, server_model
    # # args["client_model_2"] = cm2

    device = init_training_device(process_id, worker_number - 1, args.gpu_num_per_server)
    args["device"] = "cpu"

    dataset = datasetFactory(args).factory()  # loader data and partition method

    train_data_num, train_data_global, test_data_global, local_data_num, \
    train_data_local, test_data_local, class_num = dataset.load_partition_data(process_id)  # 这里的4是process Id
    # args["trainloader"] = train_data_local
    # args["testloader"] = test_data_local
    # args["train_data_num"] = train_data_num
    # args["train_data_global"] = train_data_global
    # args["test_data_global"] = test_data_global
    # args["local_data_num"] = local_data_num
    # args["class_num"] = class_num
    log = Log("main", args)
    # log.info("{}".format(train_data_num))

    # str_process_name = "SplitNN (distributed):" + str(process_id)
    # setproctitle.setproctitle(str_process_name
    SplitNN_distributed(process_id, args)
