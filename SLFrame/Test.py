import torch
import torch.nn as nn

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim

# import setproctitle

from core.log.Log import Log
from core.dataset.datasetFactory import datasetFactory
from Parse.parseFactory import parseFactory, JSON, YAML
from core.model.cnn import CNN_OriginalFedAvg, Net
from core.model.models import LeNetComplete, LeNetClientNetwork, LeNetServerNetwork
from core.client.client import SplitNNClient
from core.server.server import SplitNNServer
from core.dataset.controller.cifar10Controller import cifar10Controller
from core.splitApiO import SplitNN_distributed, SplitNN_init

log = Log("Test.py")

model = Net()
client_model = LeNetClientNetwork()
server_model = LeNetServerNetwork()
# client_model = nn.Sequential(*nn.ModuleList(model.children())[:split_layer])
# server_model = nn.Sequential(*nn.ModuleList(model.children())[split_layer:])
print(type(model))
print(client_model)
print(server_model)


def train(epoch, client):
    server.train_mode()
    client.train_mode()
    # client.dataloader.
    acts, labels = client.forward_pass()
    server.forward_pass(acts, labels)
    # log.info(server.correct)
    grad = server.backward_pass()
    client.backward_pass(grad)


def test(client):
    client.eval_mode()
    server.eval_mode()
    acts, labels = client.forward_pass()
    server.forward_pass(acts, labels)

    # log.info(server.correct)
    server.validation_over()


def init_training_device(process_ID, fl_worker_num, gpu_num_per_machine):
    # initialize the mapping from process ID to GPU ID: <process ID, GPU ID>
    logging = Log("init_training_device")
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
    d = {
        "dataset": "mnist",
        "dataDir": "./data/mnist",
        'download': True,
        'partition_method': 'homo',
        'log_step': 50,
        "rank": 1,
        "max_rank": 2,
        "lr": 0.01,
        "epochs": 2,
        "server_rank": 1,
        'device': 'cpu',

    }
    args.save(d, './config.yaml')
    d = args.load('./config.yaml')
    # comm, process_id, worker_number = SplitNN_init(parse=args)
    comm, process_id, worker_number = SplitNN_init()
    args["comm"] = comm
    args["process_id"] = process_id
    args["worker_number"] = worker_number

    args["client_model"] = client_model
    args["server_model"] = server_model

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
    dataset = datasetFactory().factory(args)  # loader data and partition method
    # print(dataset)

    train_data_num, train_data_global, test_data_global, local_data_num, \
    train_data_local, test_data_local, class_num = dataset.load_partition_data(4)  # 这里的4是process Id
    args["trainloader"] = train_data_local
    args["testloader"] = test_data_local

    # str_process_name = "SplitNN (distributed):" + str(process_id)
    # setproctitle.setproctitle(str_process_name)

    clientList = []
    for i in range(16):
        # log.info("{}: {}".format(i, args["trainloader"]))
        clientList.append(SplitNNClient(args))

    server = SplitNNServer(args)
    SplitNN_distributed(process_id, worker_number, "cpu", comm, client_model,
                        server_model, train_data_num, train_data_global, test_data_global,
                        local_data_num, train_data_local, test_data_local, args)
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
    """
    INFO:root:2021-12-26 17:05:56-----SplitNNServer
        phase=train acc=0.890625 loss=0.3596789836883545 epoch=25 and step=0
    INFO:root:2021-12-26 17:05:56-----SplitNNServer
        phase=train acc=0.9375 loss=0.20334549248218536 epoch=25 and step=0
    """
