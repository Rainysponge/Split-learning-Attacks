import torch
import torch.nn as nn
import torch.optim as optim
import logging
from ...log.Log import Log

from collections import Iterator


class SplitNNClient():

    def __init__(self, args):
        self.comm = args["comm"]
        self.model = args["client_model"]
        self.rank = args["rank"]
        self.MAX_RANK = args["max_rank"]
        self.SERVER_RANK = args["server_rank"]

        self.trainloader = args["trainloader"]
        self.testloader = args["testloader"]
        self.optimizer = optim.SGD(self.model.parameters(), args["lr"], momentum=0.9,
                                   weight_decay=5e-4)
        self.criterion = nn.BCEWithLogitsLoss()

        self.device = args["device"]
        self.phase = "train"
        self.epoch_count = 0
        self.batch_idx = 0
        self.MAX_EPOCH_PER_NODE = args["epochs"]
        self.scheduler = None

        self.log = Log(self.__class__.__name__, args)
        # 经过多少步就记录一次log
        self.log_step = args["log_step"] if args["log_step"] else 50
        self.args = args

    def reset_local_params(self):
        self.total = 0
        self.correct = 0
        self.val_loss = 0
        self.step = 0
        self.batch_idx = 0

    def write_log(self):
        if (self.phase == "train" and self.step % self.log_step == 0) or self.phase == "validation":
            self.log.info("phase={} acc={} loss={} epoch={} and step={}"
                          .format(self.phase, self.correct / self.total, self.val_loss, self.epoch_count, self.step))

    def forward_pass(self):
        # logging.info("{} begin run_forward_pass1".format(self.rank))
        inputs, labels = next(self.dataloader)
        # self.log.info("before acts:{}  labels:{}".format(type(inputs), labels.shape))
        # inputs, labels = inputs.to("cpu"), labels.to("cpu")
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()

        self.acts = self.model(inputs)
        # if isinstance(self.acts, tuple):
        # self.log.info("after acts:{}  d:{}".format(self.acts[0].shape, len(d)))

        if self.args["save_acts_step"] > 0 and self.step == 1 and self.phase == "validation" and self.epoch_count % self.args["save_acts_step"] == 0:
            a = self.acts
            a = a.cpu().detach()
            a = a.numpy()
            f = open("./model_save/acts/PSL/A_{}_E_{}_C_{}.txt".format(
                self.args["partition_alpha"], self.epoch_count, self.rank), "w")

            for i in range(a.shape[0]):
                f.write("{} {} {}\n".format(
                    self.rank, labels[i], str(list(a[i].flatten()))))

        if self.args["save_attack_acts_step"] > 0 and self.step <= 50 and self.phase == "validation" and self.epoch_count % self.args["save_acts_step"] == 0:
            a = self.acts
            a = a.cpu().detach()
            a = a.numpy()
            f = open("./model_save/attack_acts/PSL/A_{}_E_{}_C_{}.txt".format(
                self.args["partition_alpha"], self.epoch_count, self.rank), "w")

            for i in range(a.shape[0]):
                f.write("{} {} {}\n".format(
                    self.rank, labels[i], str(list(a[i].flatten()))))

        return self.acts, labels

    def backward_pass(self, grads):
        # self
        # logging.info("{} begin backward_pass".format(self.rank))
        self.acts.backward(grads)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        # logging.info("{} end backward_pass".format(self.rank))

    """
    如果模型有dropout或者BN层的时候，需要改变模型的模式
    """

    def eval_mode(self):
        if not isinstance(self.testloader, Iterator):
            self.dataloader = iter(self.testloader)
        else:
            self.dataloader = self.testloader
        self.phase = "validation"
        self.model.eval()
        self.reset_local_params()

    def train_mode(self):
        if not isinstance(self.trainloader, Iterator):
            self.dataloader = iter(self.trainloader)
            logging.info(isinstance(self.trainloader, Iterator))
        else:
            self.dataloader = self.trainloader
            logging.info(isinstance(self.trainloader, Iterator))
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=1e-3, steps_per_epoch=len(self.dataloader), epochs=15
            )

        self.phase = "train"
        self.model.train()
        self.reset_local_params()

    def print_com_size(self, com_manager):
        self.log.info("worker_num={} phase={} epoch_send={} epoch_receive={} total_send={} total_receive={}"
                      .format(self.rank, self.phase, com_manager.send_thread.tmp_send_size,
                              com_manager.receive_thread.tmp_receive_size,
                              com_manager.send_thread.total_send_size, com_manager.receive_thread.total_receive_size))
