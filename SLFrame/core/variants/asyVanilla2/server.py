from ...log.Log import Log
import torch
import torch.nn as nn
import logging
import torch.optim as optim
import sys

sys.path.extend("../../../")


class SplitNNServer():
    def __init__(self, args):
        self.log = Log(self.__class__.__name__, args)
        self.args = args
        self.comm = args["comm"]
        self.model = args["server_model"]
        self.MAX_RANK = args["max_rank"]

        self.validation_sign_number = 0

        self.epoch = 0
        # 经过多少步就记录一次log
        self.log_step = args["log_step"] if args["log_step"] else 50
        self.train_mode()
        self.optimizer = optim.SGD(self.model.parameters(), args["lr"], momentum=0.9,
                                   weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        # self.criterion = nn.BCEWithLogitsLoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        # self.optimizer = optim.SGD(self.model.parameters(), args["lr"], momentum=0.9,
        #                            weight_decay=5e-4)
        # self.criterion = nn.CrossEntropyLoss()

    def train_mode(self):
        self.model.train()
        self.reset_local_params()
        self.phase = "train"

    def eval_mode(self):
        self.model.eval()
        self.reset_local_params()
        self.phase = "validation"

    def forward_pass(self, acts, labels):
        self.acts = acts
        self.optimizer.zero_grad()
        self.acts.retain_grad()

        logits = self.model(acts)
        _, predictions = logits.max(1)
        self.loss = self.criterion(logits, labels)
        self.total = labels.size(0)
        self.correct = predictions.eq(labels).sum().item()
        self.val_loss = self.loss.item()

    def backward_pass(self):
        # self.loss.backward(retain_graph=True)

        self.loss.backward(retain_graph=True)
        # self.log.info("self.loss.backward()")

        self.optimizer.step()
        # self.log.info("self.optimizer.step()")
        #     self.optimizer, max_lr=1e-3, steps_per_epoch=len(self.dataloader), epochs=15
        # )
        # self.log.info(self.acts.grad.shape)

        return self.acts.grad

    def reset_local_params(self):
        self.total = 0
        self.correct = 0
        self.val_loss = 0
        self.step = 0
        self.batch_idx = 0
