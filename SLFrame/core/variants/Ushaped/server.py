import logging

import torch
import torch.nn as nn
import torch.optim as optim
import sys

sys.path.extend("../../../")

from ...log.Log import Log

class SplitNNServer():
    def __init__(self, args):
        self.log = Log(self.__class__.__name__, args)
        self.args = args
        self.comm = args["comm"]
        self.model = args["server_model"]
        self.MAX_RANK = args["max_rank"]

        self.epoch = 0
        self.log_step = args["log_step"] if args["log_step"] else 50  # 经过多少步就记录一次log
        self.active_node = 1
        self.train_mode()
        self.optimizer = optim.SGD(self.model.parameters(), args["lr"], momentum=0.9,
                                   weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()



    def train_mode(self):
        self.model.train()
        self.phase = "train"

    def eval_mode(self):
        self.model.eval()
        self.phase = "validation"

    def forward_pass(self, acts):
        self.acts=acts
        self.acts.retain_grad()
        self.optimizer.zero_grad()
        self.acts2=self.model(acts)
        return self.acts2

    def backward_pass(self,grad):
        self.acts2.backward(grad)
        self.optimizer.step()
        return self.acts.grad

    def validation_over(self):
        self.active_node = (self.active_node % self.MAX_RANK) + 1
        self.train_mode()

