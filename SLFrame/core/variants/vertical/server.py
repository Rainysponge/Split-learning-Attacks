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
        self.client_number = args["client_number"]

        self.epoch = 0
        self.log_step = args["log_step"] if args["log_step"] else 50  # 经过多少步就记录一次log
        self.train_mode()
        self.optimizer = optim.SGD(self.model.parameters(), args["lr"], momentum=0.9,
                                   weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()

        self.client_train_logits_list = dict()
        # self.host_local_test_logits_list = dict()

        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.client_number):
            self.flag_client_model_uploaded_dict[idx] = False

        self.loss_list = list()

    def train_mode(self):
        self.model.train()
        self.phase = "train"

    def eval_mode(self):
        self.model.eval()
        self.phase = "validation"

    def forward_pass(self, acts, labels):
        # self.acts = acts
        acts_list = []
        for idx in range(1, self.client_number+1):
            acts_list.append(self.client_train_logits_list[idx])
        self.acts = torch.cat(acts_list, dim=1)
        self.optimizer.zero_grad()
        self.acts.retain_grad()
        logits = self.model(self.acts)
        _, predictions = logits.max(1)
        self.loss = self.criterion(logits, labels)
        self.total = labels.size(0)
        self.correct = predictions.eq(labels).sum().item()
        self.val_loss = self.loss.item()

    def backward_pass(self):
        self.loss.backward(retain_graph=True)
        self.optimizer.step()
        return self.acts.grad

    def check_whether_all_receive(self):
        self.log.info(self.client_number)
        for idx in range(1, self.client_number+1):
            if idx not in self.flag_client_model_uploaded_dict or not self.flag_client_model_uploaded_dict[idx]:
                # self.log.info("idx: {}".format(idx))
                return False
        for idx in range(1, self.client_number+1):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def add_client_local_result(self, index, host_train_logits):
        # logging.info("add_client_local_result. index = %d" % index)
        self.client_train_logits_list[index] = host_train_logits
        self.flag_client_model_uploaded_dict[index] = True
