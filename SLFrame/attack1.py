import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import logging
import numpy
import random

from core.variants.attacks.model_inversion_attack_against_CL.white_box import white_box_inversion
from Parse.parseFactory import parseFactory, JSON, YAML
from core.log.Log import Log
from core.dataset.datasetFactory import datasetFactory


args = parseFactory(fileType=YAML).factory()
args.load('./config.yaml')
if args.seed != -1:
    numpy.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
args["client_number"] = 10
args["batch_size"] = 16
model_path = "./model_save/attack_acts/PSL/{}/".format(args["dataset"]) 

dataset = datasetFactory(args).factory()
train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num = dataset.load_partition_data(
    1)

target_model = torch.load(model_path+"C_1_A_2_E_100.pkl")
target_model = target_model.cuda()
test_iter = iter(test_data_local)
nm = 0
while nm < len(test_data_local):
    nm += 1
    inputs, label = test_iter.next()
    inputs, label = inputs.cuda(), label.cuda()

    torchvision.utils.save_image(
        inputs, model_path+"pics/{}_ref.png".format(nm))
    # print(label[i])
    xGen = white_box_inversion(target_model, target_model(
        inputs), inputs, nm, model_path, iterN=500000, lr=0.005,lambda_TV=0.0001,lambda_l2=0)
    torchvision.utils.save_image(
        xGen, model_path+"pics/{}_inv.png".format(nm))
    # torchvision.utils.save_image(
    #     xxGen, model_path+"pics/{}_inv_2.png".format(nm))
    if nm > 5:
        break
