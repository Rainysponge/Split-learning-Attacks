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

        self.rank = args["rank"]

        self.model_param_dict = dict()
        self.client_sample_dict = dict()
        self.acts_dict = dict()

        self.acts_num = 0
        self.model_param_num = 0
        self.sum_sample_number = 0

        self.epoch = 0
        self.log_step = args["log_step"] if args["log_step"] else 50  # 经过多少步就记录一次log
        self.train_mode()
        self.criterion = nn.CrossEntropyLoss()

    def train_mode(self):
#        self.model.train()
        self.phase = "train"

    def eval_mode(self):
     #   self.model.eval()
        self.phase = "validation"


