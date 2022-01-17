import torch
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
from core.model.cnn import CNN_OriginalFedAvg, Net
from core.model.models_for_U import LeNetClientNetworkPart1,LeNetClientNetworkPart2, LeNetServerNetwork
from core.variants.Ushaped.client import SplitNNClient
from core.variants.Ushaped.server import SplitNNServer
from core.splitApi import SplitNN_distributed, SplitNN_init

# log = Log("Test.py")

client_model = LeNetClientNetworkPart1()
client_model_2=LeNetClientNetworkPart2()
server_model = LeNetServerNetwork()
# client_model = nn.Sequential(*nn.ModuleList(model.children())[:split_layer])
# server_model = nn.Sequential(*nn.ModuleList(model.children())[split_layer:])


if __name__ == '__main__':
    args = parseFactory(fileType=YAML).factory()
    """
        解析器展示, 把需要的数据放在d里面，或者在下面的args里面 eg：args["model"] = client_model
        这里写过一次之后就可以注释掉了，存在文件中了
    """
    d = {
        "dataset": "mnist",
        "dataDir": "./data/mnist",
        'download': True,
        'partition_method': 'homo',
        'log_step': 20,
        "rank": 1,
        "max_rank": 2,
        "lr": 0.01,
        "epochs": 2,
        "server_rank": 0,
        'device': 'cpu',

    }
    args.save(d, './config.yaml')
    d = args.load('./config.yaml')
    # comm, process_id, worker_number = SplitNN_init(parse=args)
    comm, process_id, worker_number = SplitNN_init(args)
    args["rank"] = process_id  # 设置当前process_id

    args["client_model"] = client_model
    args["client_model_2"] = client_model_2
    args["server_model"] = server_model
    log = Log("Test.py", args)
    """
        下面相当于   对于通讯而言可以不用改 
    if args.dataset == "cifar10":
        data_loader = load_partition_data_distributed_cifar10
    elif args.dataset == "cifar100":
        data_loader = load_partition_data_distributed_cifar100
    elif args.dataset == "cinic10":
        data_loader = load_partition_data_distributed_cinic10
    # elif args.dataset == 'MNIST':
    #     data_loader = load_partition_data_mnist
    else:
        data_loader = load_partition_data_distributed_cifar10

    train_data_num, train_data_global, \
    test_data_global, local_data_num, \
    train_data_local, test_data_local, class_num = data_loader(process_id, args.dataset, args.data_dir,
                                                               args.partition_method, args.partition_alpha,
                                                               args.client_number, args.batch_size)
    """
    dataset = datasetFactory(args).factory()  # loader data and partition method
    # print(dataset)

    train_data_num, train_data_global, test_data_global, local_data_num, \
    train_data_local, test_data_local, class_num = dataset.load_partition_data(process_id)  # 这里的4是process Id
    args["trainloader"] = train_data_local
    args["testloader"] = test_data_local
    args["train_data_num"] = train_data_num
    args["train_data_global"] = train_data_global
    args["test_data_global"] = test_data_global
    args["local_data_num"] = local_data_num
    args["class_num"] = class_num



    # str_process_name = "SplitNN (distributed):" + str(process_id)
    # setproctitle.setproctitle(str_process_name)

    SplitNN_distributed(process_id, args)
    # train(0, clientList[0])
    # torch.save(clientList[0].model, 'checkpoint.pth.tar')
    # for i in range(25):
    #     # 这里是单个client的测试， 为了测试把server.py里面57行改为的self.loss.backward(retain_graph=True)
    #     # 原本是没有retain_graph=True
    #     for j in range(16):
    #         clientList[j].model = (torch.load('checkpoint.pth.tar'))
    #         train(i, clientList[j])
    #
    #     test(clientList[8])

