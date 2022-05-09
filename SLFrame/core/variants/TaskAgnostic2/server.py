import torch
import torch.nn as nn
import torch.optim as optim
import sys
import queue

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
        self.client_model_uploaded_number = 0
        self.sum_sample_number = 0
        self.model_param_num = 0
        self.model_ready_num = 0
        self.message_queue = queue.Queue()
        self.client_act_check_dict = dict()
        self.model_param_dict = dict()
        self.client_act_dict = dict()
        # self.sender_list = dict()

        self.epoch = 0
        self.log_step = args["log_step"] if args["log_step"] else 50  # 经过多少步就记录一次log
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
        self.acts = acts
        # self.log.info(len(acts))
        # self.log.info(acts[0])
        self.acts.retain_grad()
        self.optimizer.zero_grad()
        self.acts2 = self.model(acts)
        return self.acts2


    def backward_pass(self, grad):
        self.acts2.backward(grad, retain_graph=True)
        self.optimizer.step()
        return self.acts.grad

    def check_whether_all_receive(self):
        # self.log.info("self.client_model_uploaded_number, self.client_number :{}, {}".format(
        #     self.client_model_uploaded_number, self.client_number))
        if self.client_model_uploaded_number == self.client_number - 1:
            self.client_model_uploaded_number = 0

            return True
        return False


