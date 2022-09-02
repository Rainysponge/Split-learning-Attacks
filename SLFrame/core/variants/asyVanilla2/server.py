import torch
import torch.nn as nn
import logging
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

        self.validation_sign_number = 0

        self.epoch = 0
        self.log_step = args["log_step"] if args["log_step"] else 50  # 经过多少步就记录一次log
        self.train_mode()
        # self.optimizer = optim.SGD(self.model.parameters(), args["lr"], momentum=0.9,
        #                            weight_decay=5e-4)
        # self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        # self.criterion = nn.BCEWithLogitsLoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=1e-3, steps_per_epoch=300, epochs=30
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def train_mode(self):
        self.model.train()
        self.phase = "train"

    def eval_mode(self):
        self.model.eval()
        self.reset_local_params()
        self.phase = "validation"

    def forward_pass(self, acts, labels):
        self.acts = acts
        self.optimizer.zero_grad()
        self.acts.edge_attr.retain_grad()
        # self.log.info(acts.x.shape)
        if isinstance(acts, tuple):
            self.log.info("acts[0]:{}  acts[1]:{}".format(acts[0].shape, type(acts[1])))
            logits = self.model(acts[0], acts[1])
            self.log.info("logits:{}  acts[1]:{}".format(type(logits), type(acts[1])))
        else:
            # self.log.info(acts.x.shape)
            logits = self.model(acts)
            # self.log.info("acts.x.shape: {}".format(acts.x.shape))
        # self.log.info("acts:{}  labels:{}".format(type(logits), labels.shape))
        _, predictions = logits.max(1)
        batch_preds = torch.sigmoid(logits)
        logging.info("predictions:{}  labels:{}".format(batch_preds, labels.shape))
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
        logging.info("self.acts.edge_attr.grad: {}".format(self.acts.edge_attr.grad))
        return self.acts.edge_attr.grad

    def reset_local_params(self):
        self.total = 0
        self.correct = 0
        self.val_loss = 0
        self.step = 0
        self.batch_idx = 0
