import torch
import torch.nn as nn
import torch.optim as optim
import sys
import random

sys.path.extend("../../../")

from ...log.Log import Log


class SplitNNServer():
    def __init__(self, args):
        self.log = Log(self.__class__.__name__, args)
        self.args = args
        self.comm = args["comm"]
        self.model = args["server_model"]
        self.MAX_RANK = args["max_rank"]

        self.client_number = args["client_number"]

        self.epoch = 0
        self.log_step = args["log_step"] if args["log_step"] else 50  # 经过多少步就记录一次log
        self.train_mode()
        self.client_model_uploaded_number = 0
        self.client_train_grads_list = dict()
        self.loss_list = list()
        self.act_dict = dict()
        self.res_dict = dict()
        self.act_number = 0

        self.total = 0
        self.correct = 0
        self.val_loss = 0
        self.step = 0
        self.batch_idx = 0
        # self.optimizer = optim.SGD(self.model.parameters(), args["lr"], momentum=0.9,
        #                            weight_decay=5e-4)
        # self.optimizer = torch.optim.Adam(self.model.parameters(),
        #                                   lr=args["lr"] * (self.client_number ** args["SGLR_splitLR_alpha"]),
        #                                   betas=(0.9, 0.999),
        #                                   eps=1e-08,
        #                                   weight_decay=0,
        #                                   amsgrad=False)

        # self.optimizer = self.splitLr()
        self.optimizer = optim.SGD(self.model.parameters(),
                                   args["lr"] * (self.client_number ** args["SGLR_splitLR_alpha"]),
                                   momentum=0.9, weight_decay=5e-4)

        # if "SGLR_splitLR" not in args or not args["SGLR_splitLR"]:
        #     self.log.info("SGLR_splitLR: {}".format(args["SGLR_splitLR"]))
        #     self.optimizer = optim.SGD(self.model.parameters(), args["lr"], momentum=0.9, weight_decay=5e-4)
        # else:
        #     self.log.info("SGLR_splitLR: {}".format(args["SGLR_splitLR"]))
        #     self.optimizer = optim.SGD(self.model.parameters(),
        #                                args["lr"] * (self.client_number ** args["SGLR_splitLR_alpha"]),
        #                                momentum=0.9, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()

    def train_mode(self):
        self.model.train()
        self.reset_local_params()
        self.phase = "train"

    def eval_mode(self):
        self.model.eval()
        self.reset_local_params()
        self.phase = "validation"

    def forward_pass(self, acts, labels, sender):
        self.acts = acts
        if self.phase == "validation":
            self.optimizer.zero_grad()
        self.acts.retain_grad()

        logits = self.model(acts)
        _, predictions = logits.max(1)
        self.loss = self.criterion(logits, labels)
        self.total = labels.size(0)
        self.correct = predictions.eq(labels).sum().item()
        self.val_loss = self.loss.item()
        self.res_dict[sender] = (self.total, self.correct, self.val_loss)
        if self.phase == "validation":
            self.optimizer.zero_grad()

    def backward_pass(self):
        self.loss.backward(retain_graph=True)
        # self.optimizer.step()
        # self.log.info(self.acts.grad.shape)
        return self.acts.grad

    def check_whether_all_receive(self):
        # self.log.info("self.client_model_uploaded_number, self.client_number :{}, {}".format(
        #     self.client_model_uploaded_number, self.client_number))
        if self.client_model_uploaded_number == self.client_number:
            self.client_model_uploaded_number = 0
            active_list = random.sample([i for i in range(1, self.client_number + 1)],
                                        round(self.client_number * self.args["SGLR_splitAvg_active"]))

            return True, active_list
        return False, []

    def add_client_local_grads(self, index, host_train_grads):
        # logging.info("add_client_local_result. index = %d" % index)
        self.client_train_grads_list[index] = host_train_grads
        self.client_model_uploaded_number += 1

    def splitAvg(self, active_list):
        # self.args["client_number"]
        sum = 0
        # self.log.info("self.client_number : {}".format(self.client_number))
        for index in active_list:
            sum += self.client_train_grads_list[index]
        return sum / len(active_list)

    def splitLr(self):
        args = self.args
        if "SGLR_splitLR" not in args or not args["SGLR_splitLR"]:
            self.log.info("SGLR_splitLR: {}".format(args["SGLR_splitLR"]))
            return optim.SGD(self.model.parameters(), args["lr"], momentum=0.9, weight_decay=5e-4)
        else:
            self.log.info("SGLR_splitLR: {}".format(args["SGLR_splitLR"]))
            return optim.SGD(self.model.parameters(), args["lr"] * (self.client_number ** args["SGLR_splitLR_alpha"]),
                             momentum=0.9, weight_decay=5e-4)

    def reset_local_params(self):
        self.total = 0
        self.correct = 0
        self.val_loss = 0
        self.step = 0
        self.batch_idx = 0
