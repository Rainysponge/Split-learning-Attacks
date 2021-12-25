import torch
import torch.nn as nn

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim

from core.log.Log import Log
from core.dataset.datasetFactory import datasetFactory
from Parse.parseFactory import parseFactory, JSON, YAML
from core.model.cnn import CNN_OriginalFedAvg, Net
from core.model.models import LeNetComplete, LeNetClientNetwork, LeNetServerNetwork
from core.client.client import SplitNNClient
from core.server.server import SplitNNServer

# log = Log("Test.py")
# log.info("Test")
#
# 单个Client 后面和通信息模块对接上流程就跑通了

# parse = parseFactory(fileType=YAML).factory()
# d = {
#     "model": [i for i in range(7)],
#     "dataset": "cifar10",
#     "dataDir": "./data/cifar10",
#     'download': True,
#     'partition_method': 'homo'
#
# }
# parse.save(d, './config.yaml')
# d = parse.load('./config.yaml')
# dataset = datasetFactory().factory(parse)  # loader data and partition method
# print(dataset)
# dataset.partition_data()

log = Log("Test.py")

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data/mnist/',
                               train=True,
                               download=True,
                               transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64)

test_dataset = datasets.MNIST(root='./data/mnist/',
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False, batch_size=64)

split_layer = 1
model = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

client_model = LeNetClientNetwork()
server_model = LeNetServerNetwork()
# client_model = nn.Sequential(*nn.ModuleList(model.children())[:split_layer])
# server_model = nn.Sequential(*nn.ModuleList(model.children())[split_layer:])
print(type(model))
print(client_model)
print(server_model)

# model = client_model
args = {}
args["comm"] = None
args["model"] = client_model
args["trainloader"] = train_loader
args["testloader"] = test_loader
args["rank"] = 1
args["max_rank"] = 2
args["lr"] = 0.01
args["epochs"] = 2
args["server_rank"] = 1
args['device'] = 'cpu'
client = SplitNNClient(args)
# print(client)
args["comm"] = None
args["model"] = server_model
args["max_rank"] = 4

args["log_step"] = 50

server = SplitNNServer(args)


def train(epoch):
    server.train_mode()
    client.train_mode()
    # client.dataloader.
    acts, labels = client.forward_pass()
    server.forward_pass(acts, labels)
    # log.info(server.correct)
    grad = server.backward_pass()
    client.backward_pass(grad)


def test():
    client.eval_mode()
    server.eval_mode()
    acts, labels = client.forward_pass()
    server.forward_pass(acts, labels)

    log.info(server.correct)
    server.validation_over()


if __name__ == '__main__':
    for epoch in range(300):
        train(epoch)
        test()
